# Deep Reinforcement Learning for *De Novo* Synthesis of eye drops
In this work, for the first time we present a pipeline generating potential active substances in eye drops based on protein-ligand interaction with predefined properties. For that, we trained classical ML models to predict corneal permeability, binding of the drug to melanin and eye irritation. We further constructed a novel reward function to explicitly take into account the desired properties of eye drops. Using our pipeline we generate 1762 unique molecules with improved properties characterizing active substances of eye drops.

![Pipeline for *de novo* synthesis of eye drops](Images/Pipeline.png)
---
## Setup Python environment
```
conda env create -f environment.yml
conda activate freedpp
```
We used KAN model for melanin binding prediction. You can install pykan environment from [here](https://github.com/KindXiaoming/pykan)
## Dependency

| Package | Version | 
|:----------------:|:---------:|
| Python | 3.7.12 | 
| PyTorch | 1.12.1 |
| TorchVision | 0.13.1 |
| CUDA | 11.3.1 |
| DGL | 0.9.1.post1 |
| RDKit | 2020.09.1.0 |

## Usage
Run these commands from freedpp/freedpp folder.
### Training
```
python main.py --exp_root ../experiments
 --alert_collections ../alert_collections.csv     --fragments ../zinc_crem.json
 --receptor ../COX-2.pdbqt     --vina_program ./env/qvina02     --starting_smile "O=C(O)C(*)c1c(*)c(*)c(*)c(*)c1(*)"
 --fragmentation crem     --num_sub_proc 12     --n_conf 1     --exhaustiveness 1     --save_freq 10
--epochs 200     --commands "train,sample"     --reward_version soft     --box_center "27.116 24.090 14.936"
--box_size "9.427,10.664,10.533"     --seed 150     --name freedpp
 --objectives "DockingScore,Corneal,Melanin,Irritation"     --weights "1.0,1.0,1.0,1.0" --num_mols 5000   
```
### Evaluating
```
python main.py --exp_root ../experiments
 --alert_collections ../alert_collections.csv     --fragments ../zinc_crem.json
 --receptor ../COX-2.pdbqt     --vina_program ./env/qvina02     --starting_smile "O=C(O)C(*)c1c(*)c(*)c(*)c(*)c1(*)"
 --fragmentation crem     --num_sub_proc 12     --n_conf 1     --exhaustiveness 1     --save_freq 10
--epochs 200     --commands "evaluate"     --reward_version soft     --box_center "27.116 24.090 14.936"
--box_size "9.427,10.664,10.533"     --seed 150     --name freedpp
 --objectives "DockingScore,Corneal,Melanin,Irritation"     --weights "1.0,1.0,1.0,1.0" --checkpoint ..experiments/freedpp/ckpt/model_200.pth  
```
## Contents
* Data
  * Contains .csv files with datasets including SMILES and target value
* Scripts
  * Contains .ipynb files with code for running ML models
