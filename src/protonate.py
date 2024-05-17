'''
originally sourced from https://github.com/Xundrug/MolTaut/blob/master/moltaut_src/molgpka/protonate.py
'''
from predict_pka import predict
from copy import deepcopy
from rdkit import Chem

from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from itertools import combinations
import pdb

import json
import numpy as np
import random
import os
import copy

def modify_mol(mol, acid_dict, base_dict):
    '''
    Sets a property of the correlated pka and type (Acid or Basic) to each ionizable atom in a given molecule.
    Use after using predict function from predict_pka
        mol: smiles rdkit object
            (type: rdkit object)
        acid_dict: dictionary with atom index as keys and pka as value
            (type: dictionary)
        base_dict: dictionary with atom index as keys and pka as value
            (type: dictionary)
        Returns smiles rdkit object
    '''
    # get the index of each atom
    for at in mol.GetAtoms():
        idx = at.GetIdx()
        # if the index of the atom is in acid_dict keys
        if idx in set(acid_dict.keys()):
            # value = pka correlated to atom index of interest
            value = acid_dict[idx]
            # atom of interest is H, so get first neighboring atom
            nat = at.GetNeighbors()[0]
            nat.SetProp("ionization", "A")
            nat.SetProp("pKa", str(value))
        # elif the index of the atom is in base_dict keys
        elif idx in set(base_dict.keys()):
            value = base_dict[idx]
            at.SetProp("ionization", "B")
            at.SetProp("pKa", str(value))
        else:
            at.SetProp("ionization", "O")
    # remove the explicit hydrogens in the molecule
    nmol = AllChem.RemoveHs(mol)
    return nmol

def get_pKa_data(mol, ph, tph):
    '''
    Separates the ionizable atom as either stable or unstable based on pKa and given pH
        Returns: two lists 
    '''
    stable_data, unstable_data = [], []
    for at in mol.GetAtoms():
        props = at.GetPropsAsDict()
        acid_or_basic = props.get('ionization', False)
        pKa = float(props.get('pKa', False))
        idx = at.GetIdx()
        # if property of the atom is acidic
        if acid_or_basic == "A":
            # and if pKa is less than pH, then add index of atom, pKa, and type to stable list to be deprotonated
            if pKa <= ph:
                stable_data.append( [idx, pKa, "A"] )
            # if pKa is more than pH, then add index of atom, pKa, and type to unstable list
            else:
                unstable_data.append( [idx, pKa, "A"] )
        # if property of atom is basic 
        elif acid_or_basic == "B":
            # and if pKa is more than pH, then add index of atom, pKa, and type to stable list to be protonated
            if pKa > ph + tph:
                stable_data.append( [idx, pKa, "B"] )
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append( [idx, pKa, "B"] )
        else:
            continue
    return stable_data, unstable_data

def modify_acid(at):
    '''
    Deprotonates atom
    '''
    hnum = at.GetNumExplicitHs()
    at.SetFormalCharge(-1)
    at.SetNumExplicitHs(hnum-1)
    return

def modify_base(at):
    '''
    Protonates atom
    '''
    hnum = at.GetNumExplicitHs()
    at.SetFormalCharge(1)
    at.SetNumExplicitHs(hnum+1)
    return

def modify_stable_pka(new_mol, stable_data):
    breakpoint()
    new_unsmis = []
    for pka_data in stable_data:
        idx, pka, acid_or_basic = pka_data
        at = new_mol.GetAtomWithIdx(idx)
        if acid_or_basic == "A":
            # deprotonate atom
            modify_acid(at)
        elif acid_or_basic == "B":
            # protonate atom
            modify_base(at)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
        new_unsmis.append([smi, pka])
    return new_unsmis

def modify_unstable_pka(mol, unstable_data, i):
    breakpoint()
    combine_pka_datas = list(combinations(unstable_data, i))
    new_unsmis = []
    for pka_datas in combine_pka_datas:
        # pka_datas example ([25, 5.3894997, 'B'],)
        new_mol = deepcopy(mol)
        if len(pka_datas) == 0:
            continue
        for pka_data in pka_datas:
            # pka_data example [25, 5.3894997, 'B']
            idx, pka, acid_or_basic = pka_data
            at = new_mol.GetAtomWithIdx(idx)
            if acid_or_basic == "A":
                modify_acid(at)
            elif acid_or_basic == "B":
                modify_base(at)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
        new_unsmis.append([smi, pka])
    return new_unsmis

def protonate_mol(smi, ph, tph):
    breakpoint()
    # make smiles into object to be read by rdkit
    omol = Chem.MolFromSmiles(smi)
    # run pka prediction of molecule; returns base_dict, acid_dict, and smiles object
    obase_dict, oacid_dict, omol = predict(omol)
    # get molecule object with each ionziable atom containing pka and A or B type info
    mc = modify_mol(omol, oacid_dict, obase_dict)
    stable_data, unstable_data = get_pKa_data(mc, ph, tph)
    new_smis = []
    n = len(unstable_data)
    # if based on pKa and pH rules, the deprotonated or protonated state is viable, then its in stable_data
    if n == 0:
        new_mol = deepcopy(mc)
        modify_stable_pka(new_mol, stable_data)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
        new_smis.append(smi)
    else:
        for i in range(n + 1):
            new_mol = deepcopy(mc)
            modify_stable_pka(new_mol, stable_data)
            if i == 0:
                new_smis.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))))
            new_unsmis = modify_unstable_pka(new_mol, unstable_data, i)
            new_smis.extend(new_unsmis)
    return new_smis

if __name__=="__main__":
    smi = "Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)n2)cn1"
    pt_smis= protonate_mol(smi, ph=7.4, tph=2.5)
    breakpoint()
    print(pt_smis)
    print(smile_info)

   

