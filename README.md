# data2smiles

A framework for generating candidate molecular structures (SMILES) conditioned on tabular or experimental data.  
This repository provides implementations such as conditional generative models (e.g. `_vae`), and can be extended with other approaches in the future.

---

## Installation

Make sure you have Python 3.10+ and `pip` installed. Then run:

```bash
sudo apt update
sudo apt -y install python3-pip

# Core dependencies
pip3 install "numpy<2" pandas==2.2.3 scikit-learn==1.7.2 scipy==1.15.3
pip3 install rdkit-pypi==2022.9.5

# Clean up conflicting packages
pip3 uninstall -y torch torchvision torchaudio
pip3 uninstall -y orb-models chgnet

# Install PyTorch stack
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --no-cache-dir

---

