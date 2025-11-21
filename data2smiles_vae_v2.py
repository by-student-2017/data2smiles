# ==============================
# sudo apt update
# sudo apt -y install python3-pip
# pip3 install "numpy<2" pandas==2.2.3 scikit-learn==1.7.2 scipy==1.15.3
# pip3 install rdkit-pypi==2022.9.5
# pip3 uninstall -y torch torchvision torchaudio
# pip3 uninstall -y orb-models chgnet
# pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --no-cache-dir
# ==============================

# ==============================
# Usage
# python3 data2smiles_vae_v2.py
# ==============================

import os
import sys
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

#from pprint import pprint
import pandas as pd

# ==============================
# 0. Args
# ==============================
target_adsorption = 4.0
if len(sys.argv) > 1:
    try:
        target_adsorption = float(sys.argv[1])
    except ValueError:
        print("The argument must be a number (e.g., 4.0).")
        sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==============================
# 1. Load data
# ==============================
csv_file = "data.csv"
df = pd.read_csv(csv_file)

# Required column check
required_cols = ["smiles", "target_data",
                 "metal","activation",
                 "carbonization_temp_C","surface_area_m2_g","conditions_K","conditions_bar"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Only valid SMILES to be learned with RDKit
def is_valid_smiles(s):
    try:
        m = Chem.MolFromSmiles(s)
        return m is not None
    except Exception:
        return False

df = df[df["smiles"].apply(is_valid_smiles)].reset_index(drop=True)
if len(df) < 10:
    raise ValueError("Valid SMILES rows < 10. Check data.")

# ==============================
# 2. Tokenization
# ==============================
SPECIALS = ["<PAD>", "<START>", "<END>"]
chars = sorted(set("".join(df["smiles"])))
vocab = SPECIALS + chars
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}
PAD = stoi["<PAD>"]; START = stoi["<START>"]; END = stoi["<END>"]

max_len_smiles = int(df["smiles"].apply(len).max())
max_len = max_len_smiles + 2  # START and END included

def encode_smiles(s):
    tokens = [START] + [stoi[c] for c in s] + [END]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        tokens[-1] = END
    pad_len = max_len - len(tokens)
    tokens = tokens + [PAD]*pad_len
    return np.array(tokens, dtype=np.int64)

def decode_tokens(tokens):
    out = []
    for t in tokens:
        if t == END:
            break
        if t in (PAD, START):
            continue
        out.append(itos[int(t)])
    return "".join(out)

encoded = np.stack(df["smiles"].apply(encode_smiles).values)  # [N, T]
N, T = encoded.shape
V = len(vocab)

# ==============================
# 3. Conditions / target as conditioning vector
# ==============================
cond_cols_cat = ["metal","activation"]
cond_cols_num = ["carbonization_temp_C","surface_area_m2_g","conditions_K","conditions_bar","target_data"]

# Categories are one-hot, values ​​are standardized
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cond_cols_cat),
        ("num", "passthrough", cond_cols_num)
    ],
    remainder="drop"
)
cond_mat = preprocessor.fit_transform(df[cond_cols_cat + cond_cols_num])
# Normalization (simple standardization of numerical values ​​only)
from sklearn.preprocessing import StandardScaler
num_scaler = StandardScaler()
num_idx_start = preprocessor.transformers_[0][1].n_features_in_
ohe = preprocessor.named_transformers_["cat"]
cond_cat_dim = sum(len(cats) for cats in ohe.categories_)
cond_num = cond_mat[:, cond_cat_dim:]
cond_num = num_scaler.fit_transform(cond_num)
cond = np.concatenate([cond_mat[:, :cond_cat_dim], cond_num], axis=1)
cond_dim = cond.shape[1]

# torch tensors
X_tokens = torch.tensor(encoded, dtype=torch.long, device=device)  # [N,T]
X_cond = torch.tensor(cond, dtype=torch.float32, device=device)    # [N,cond_dim]

# ==============================
# 4. Conditional VAE (GRU encoder/decoder)
# ==============================
class CVAE(nn.Module):
    def __init__(self, vocab_size, cond_dim, emb_dim=128, hid=256, z_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)

        # encoder: GRU over tokens, concatenated with condition
        self.enc_gru = nn.GRU(emb_dim, hid, batch_first=True)
        self.enc_fc_mu = nn.Linear(hid + cond_dim, z_dim)
        self.enc_fc_logvar = nn.Linear(hid + cond_dim, z_dim)

        # decoder init from (z, cond)
        self.dec_fc_init = nn.Linear(z_dim + cond_dim, hid)
        self.dec_gru = nn.GRU(emb_dim + cond_dim, hid, batch_first=True)
        self.dec_fc_out = nn.Linear(hid, vocab_size)

    def encode(self, tokens, cond):
        # tokens: [B,T], cond: [B,cond_dim]
        emb = self.emb(tokens)  # [B,T,emb]
        h_out, h_n = self.enc_gru(emb)  # h_n: [1,B,hid]
        h_last = h_n.squeeze(0)         # [B,hid]
        h_cat = torch.cat([h_last, cond], dim=1)  # [B,hid+cond]
        mu = self.enc_fc_mu(h_cat)
        logvar = self.enc_fc_logvar(h_cat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, cond, max_len, teacher_tokens=None):
        # z: [B,z], cond: [B,cond]
        B = z.size(0)
        h0 = torch.tanh(self.dec_fc_init(torch.cat([z, cond], dim=1))).unsqueeze(0)  # [1,B,hid]
        # start tokens
        inputs = torch.full((B, 1), START, dtype=torch.long, device=z.device)
        outputs = []
        hidden = h0
        for t in range(max_len):
            emb = self.emb(inputs[:, -1])  # [B,emb]
            emb_cond = torch.cat([emb, cond], dim=1).unsqueeze(1)  # [B,1,emb+cond]
            out, hidden = self.dec_gru(emb_cond, hidden)  # out: [B,1,hid]
            logits = self.dec_fc_out(out.squeeze(1))      # [B,V]
            outputs.append(logits)
            # next token: teacher forcing if provided
            if teacher_tokens is not None and t < teacher_tokens.size(1):
                next_tok = teacher_tokens[:, t]
            else:
                next_tok = torch.argmax(logits, dim=1)
            inputs = torch.cat([inputs, next_tok.unsqueeze(1)], dim=1)
            if teacher_tokens is None:
                # optional early stop: if all END predicted
                if (next_tok == END).all():
                    break
        return torch.stack(outputs, dim=1)  # [B,T,V] (T may be <= max_len)

    def forward(self, tokens, cond):
        mu, logvar = self.encode(tokens, cond)
        z = self.reparameterize(mu, logvar)
        # teacher forcing: use ground truth tokens shifted (excluding START)
        teacher = tokens[:, 1:]  # [B,T-1]
        logits = self.decode(z, cond, max_len=tokens.size(1)-1, teacher_tokens=teacher)
        return logits, mu, logvar

def vae_loss(logits, target_tokens, mu, logvar):
    # logits: [B,T,V], target_tokens: [B,T]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         target_tokens.reshape(-1),
                         ignore_index=PAD)
    # KL
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return ce + kl, ce.item(), kl.item()

# ==============================
# 5. Train CVAE
# ==============================
BATCH = 64
EPOCHS = 30  # Adjust according to the amount of data
EMB = 128; HID = 256; ZDIM = 64

model = CVAE(vocab_size=V, cond_dim=cond_dim, emb_dim=EMB, hid=HID, z_dim=ZDIM).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

# dataloader
idxs = np.arange(N)
def batches():
    np.random.shuffle(idxs)
    for i in range(0, N, BATCH):
        sl = idxs[i:i+BATCH]
        yield X_tokens[sl], X_cond[sl]

model.train()
for ep in range(1, EPOCHS+1):
    total, ce_sum, kl_sum = 0.0, 0.0, 0.0
    for toks, cond_vec in batches():
        opt.zero_grad()
        logits, mu, logvar = model(toks, cond_vec)
        target = toks[:, 1: 1+logits.size(1)]
        loss, ce_val, kl_val = vae_loss(logits, target, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
        ce_sum += ce_val
        kl_sum += kl_val
    print(f"[Epoch {ep:02d}] loss={total:.3f} CE={ce_sum:.3f} KL={kl_sum:.3f}")

# ==============================
# 6. Conditioned sampling (improved version)
# ==============================
# Constructing the condition vector: Set metal/activation to the representative value, and set the numerical value to the representative value including target.
metal_mode = df["metal"].mode().iloc[0]
activation_mode = df["activation"].mode().iloc[0]

# Creates a condition row (if specified by the user)
cond_row = pd.DataFrame([{
    "metal": 'Cu',
    "activation": 'Steam',
    "carbonization_temp_C": 800.0,
    "surface_area_m2_g": 3400.0,
    "conditions_K": 77.0,
    "conditions_bar": 1.0,
    "target_data": 4.0
}])

# Create a Criteria Row
# If user specified, use that value; if not, use the typical value.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--metal", type=str, default=None)
parser.add_argument("--activation", type=str, default=None)
parser.add_argument("--temp", type=float, default=None)
parser.add_argument("--area", type=float, default=None)
parser.add_argument("--K", type=float, default=None)
parser.add_argument("--bar", type=float, default=None)
parser.add_argument("--ads", type=float, default=None)
args = parser.parse_args()
cond_row = pd.DataFrame([{
    "metal": args.metal if args.metal else df["metal"].mode().iloc[0],
    "activation": args.activation if args.activation else df["activation"].mode().iloc[0],
    "carbonization_temp_C": args.temp if args.temp else float(df["carbonization_temp_C"].median()),
    "surface_area_m2_g": args.area if args.area else float(df["surface_area_m2_g"].median()),
    "conditions_K": args.K if args.K else float(df["conditions_K"].median()),
    "conditions_bar": args.bar if args.bar else float(df["conditions_bar"].median()),
    "target_data": args.ads if args.ads else float(target_adsorption)
}])

model.eval()

def sample_smiles(num_samples=2000, temperature=1.8, top_k=30, top_p=0.9, jitter_eps=0.3):
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            # --- Shake the condition vector a little ---
            cond_np = cond_row.copy()
            cond_np.loc[0, "target_data"] = float(target_adsorption + np.random.uniform(-jitter_eps, jitter_eps))
            cond_np.loc[0, "surface_area_m2_g"] *= (1.0 + np.random.uniform(-0.05, 0.05))
            cond_np.loc[0, "carbonization_temp_C"] *= (1.0 + np.random.uniform(-0.05, 0.05))

            cond_vec = preprocessor.transform(cond_np)
            cond_cat = cond_vec[:, :cond_cat_dim]
            cond_num_raw = cond_vec[:, cond_cat_dim:]
            cond_num_std = num_scaler.transform(cond_num_raw)
            cond_final = np.concatenate([cond_cat, cond_num_std], axis=1)
            cond_local = torch.tensor(cond_final, dtype=torch.float32, device=device)

            # --- Widely search for latent vectors ---
            z = torch.randn((1, ZDIM), device=device) * np.random.uniform(0.8, 2.0)

            logits = model.decode(z, cond_local, max_len=max_len-1, teacher_tokens=None)
            probs = F.softmax(logits.squeeze(0) / temperature, dim=-1)

            toks = []
            for t in range(probs.size(0)):
                p = probs[t]

                # top-k filter
                if top_k is not None:
                    k = min(top_k, p.size(0))  # Limit the number of words to be less than the number of words
                    top_idx = torch.topk(p, k).indices
                    mask = torch.zeros_like(p)
                    mask[top_idx] = p[top_idx]
                    p = mask / mask.sum()
                
                # top-p (nucleus) filter
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(p, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=0)
                    mask = cum_probs <= top_p
                    if mask.sum() > 0:
                        idx = sorted_idx[mask]
                        p_masked = torch.zeros_like(p)
                        p_masked[idx] = p[idx]
                        p = p_masked / p_masked.sum()

                tok = torch.multinomial(p, 1).item()
                toks.append(tok)

            s = decode_tokens(toks)
            if s and Chem.MolFromSmiles(s):
                samples.append(s)

    samples = list(dict.fromkeys(samples))  # Duplicate removal
    return samples

# Improved Call
TOPK = 5
generated = sample_smiles(num_samples=2000, temperature=1.8, top_k=TOPK, top_p=0.9, jitter_eps=0.3)

# ==============================
# 7. Lightweight property model for ranking
# ==============================
# A lightweight model (BayesianRidge) for estimating y using existing data
# SMILES → Character index (identical tokenization)
sm_cols = [f"tok_{i}" for i in range(max_len)]
def smiles_to_fixed_tokens(s):
    return encode_smiles(s)

tok_mat = np.stack(df["smiles"].apply(smiles_to_fixed_tokens).values)
tok_df = pd.DataFrame(tok_mat, columns=sm_cols)

feat_df = pd.concat([
    df[["metal","activation","carbonization_temp_C","surface_area_m2_g","conditions_K","conditions_bar"]],
    tok_df
], axis=1)

pre_rank = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["metal","activation"]),
        ("num", "passthrough", ["carbonization_temp_C","surface_area_m2_g","conditions_K","conditions_bar"])
    ],
    remainder="passthrough"
)

rank_model = Pipeline([
    ("pre", pre_rank),
    ("mdl", BayesianRidge())
])
rank_model.fit(feat_df, df["target_data"])

# Ranking generation candidates
cand_rows = []
for s in generated:
    tks = smiles_to_fixed_tokens(s)
    row = pd.DataFrame([{
        "metal": metal_mode,
        "activation": activation_mode,
        "carbonization_temp_C": float(df["carbonization_temp_C"].median()),
        "surface_area_m2_g": float(df["surface_area_m2_g"].median()),
        "conditions_K": float(df["conditions_K"].median()),
        "conditions_bar": float(df["conditions_bar"].median()),
        **{sm_cols[i]: tks[i] for i in range(max_len)}
    }])
    mu = rank_model.predict(row)[0]
    # Pseudo uncertainty (simple approximation since BayesianRidge does not have predict_std)
    # Here we use the difference variance from the mean as a simple alternative (replace with GP if necessary)
    # Note: This is a simplification, not an exact uncertainty
    cand_rows.append({
        "smiles": s,
        "predicted_target_data": float(mu),
    })

cand_df = pd.DataFrame(cand_rows)

# Bayesian candidate distribution (simple calculation of target neighborhood probability: normal approximation assumed)
# Set the neighborhood width eps
eps = 0.3
# The variance approximation uses the error variance of the entire data (simple)
mae = mean_absolute_error(df["target_data"], rank_model.predict(feat_df))
sigma_approx = max(mae, 1e-6)

def prob_within(target, mu, sigma, eps):
    upper = (target + eps - mu) / sigma
    lower = (target - eps - mu) / sigma
    return float(norm.cdf(upper) - norm.cdf(lower))

cand_df["abs_error_to_target"] = (cand_df["predicted_target_data"] - target_adsorption).abs()
cand_df["prob_within_eps"] = cand_df.apply(
    lambda r: prob_within(target_adsorption, r["predicted_target_data"], sigma_approx, eps), axis=1
)

cand_df = cand_df.sort_values(
    by=["prob_within_eps","abs_error_to_target"],
    ascending=[False, True]
).reset_index(drop=True)

# Display top candidate distribution
print(f"\n=== Candidate distribution of generated SMILES (Top {TOPK}) conditioned on target {target_adsorption:.2f} ===")
print(cand_df.head(TOPK))

# Representative values ​​are also listed
print("\n=== Specified conditions (used for generation) ===")
print(pd.DataFrame([{
    "metal": metal_mode,
    "activation": activation_mode,
    "carbonization_temp_C": float(df["carbonization_temp_C"].median()),
    "surface_area_m2_g": float(df["surface_area_m2_g"].median()),
    "conditions_K": float(df["conditions_K"].median()),
    "conditions_bar": float(df["conditions_bar"].median()),
    "target_data (conditioned)": float(target_adsorption)
}]))