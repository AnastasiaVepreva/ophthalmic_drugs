from functools import partial
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
import sklearn
import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys
import torch
import xgboost as xgb
import os
import warnings
from kan.KAN import KAN


class Reward:
    def __init__(self, property, reward, weight=1.0, preprocess=None):
        self.property = property
        self.reward = reward
        self.weight = weight
        self.preprocess = preprocess

    def __call__(self, input):
        if self.preprocess:
            input = self.preprocess(input)
        property = self.property(input)
        reward = self.weight * self.reward(property)
        return reward, property


def identity(x):
    return x


def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def get_fps(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:166]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32)  
    fps_tensor = torch.tensor(fps_array)
    return fps_tensor

def smi_to_maccs(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(mols)):
        fp = [int(bit) for bit in MACCSkeys.GenMACCSKeys(mols[i]).ToBitString()][:167]
        fps.append(fp)
    fps_array = np.array(fps, dtype=np.int32).reshape(1, -1) 
    return fps_array

def Melanin(mol):

    model = KAN(width=[166,1,1], grid=10, k=5, seed=2024)
    model.load_state_dict(torch.load('KAN_melanin.pth'))
    fps_tensor = get_fps([Chem.MolToSmiles(mol)])
    mel = model(fps_tensor)[:, 0].detach().numpy()
    return mel[0]


def Irritation(mol):
    model = pickle.load(open('pickle_model_irritation.pkl', 'rb'))
    fps = smi_to_maccs([Chem.MolToSmiles(mol)])
    irr = model.predict_proba(fps)[0][1]
    return irr


def Corneal(mol): 
    model = xgb.XGBRegressor(random_state=10)
    model.load_model('corneal.json') 
    maccs = smi_to_maccs([Chem.MolToSmiles(mol)]) 
    cor = model.predict(maccs) 
    return cor[0]

