#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize

import os.path as osp
import numpy as np
import pandas as pd
import pdb

import torch
from utils.ionization_group import get_ionization_aid
from utils.descriptor import mol2vec
from utils.net import GCNNet

root = osp.abspath(osp.dirname(__file__))

def load_model(model_file, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model= GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model

def model_pred(m2, aid, model, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def predict_acid(mol):
    model_file = osp.join(root, "../models/weight_acid.pth")
    model_acid = load_model(model_file)

    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid)
        acid_res.update({aid:apka})
    return acid_res

def predict_base(mol):
    model_file = osp.join(root, "../models/weight_base.pth")
    model_base = load_model(model_file)

    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base) 
        base_res.update({aid:bpka})
    return base_res

def predict(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    AllChem.EmbedMolecule(mol)
    return base_dict, acid_dict, mol

def predict_for_protonate(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    return base_dict, acid_dict, mol

def add_args(parser):
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help="should contain smiles and targets in same dataframe",)
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="should contain smiles and targets in same dataframe",)

def load_data(args):
    if args.data:
        pH = 7.4
        data = pd.read_csv(args.data, names=['source'])
        data[['prefix','smiles']] = data['source'].str.split(':',expand=True)
        mol = Chem.MolFromSmiles(data['smiles'].iloc[0])
        print(mol)
        base_dict, acid_dict, m = predict(mol)
        print("base:",base_dict)
        for i in data['smiles']:
            print(i)
            mol = Chem.MolFromSmiles(i)
            breakpoint()
            base_dict, acid_dict, m = predict(mol)
            for atom_i, pka in base_dict.items():
                # key = atom index and val = pKa
                atom = m.GetAtomWithIdx(atom_i)
                atom_charge = atom.GetFormalCharge()
                # if pKa > pH, then protonate the atom given the index
                breakpoint()
                if pka > pH:
                    atom.SetNumExplicitHs(atom_charge + 1)
                    modified_smiles = Chem.MolToSmiles(m)
                    print(modified_smiles)
                elif pka <= pH and atom_charge > 0:
                    atom.SetNumExplicitHs(atom_charge - 1)
                    modified_smiles = Chem.MolToSmiles(m)
                    print(modified_smiles)

            print("base:",base_dict)
            print("acid:",acid_dict)
    #print("symbol and pKa:", m.GetAtomWithIdx(13).GetSymbol())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    load_data(args)


