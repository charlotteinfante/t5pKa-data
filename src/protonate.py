'''
originally sourced from https://github.com/Xundrug/MolTaut/blob/master/moltaut_src/molgpka/protonate.py
'''
from predict_pka import predict
from copy import deepcopy
from rdkit import Chem
import pandas as pd
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

def get_pKa_data(mol, ph):
    '''
    Separates the ionizable atom as either stable or unstable based on pKa and given pH
        mol: rdkit molecule object
        ph: pH given 
            (type: str)
        Returns: six lists. The first list contains most likely to be deprotonated molecules (acids), 
        the second list contains not as likely to deprotonate (acids), the third list contains most likely to be 
        protonated molecules (bases), the fourth list contains not as likely to protoonate molecules (bases), the 
        fifth list contains all most likely to be ionized molecules (acids and bases), the last list contains the 
        less likely to be ionized molecules (acids and bases). 
    '''
    stable_data, unstable_data = [], []
    stable_acid, unstable_acid = [], []
    stable_base, unstable_base = [], []
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
                stable_acid.append( [idx, pKa, "A"] )
            # if pKa is more than pH, then add index of atom, pKa, and type to unstable list
            else:
                unstable_data.append( [idx, pKa, "A"] )
                unstable_acid.append( [idx, pKa, "A"] )
        # if property of atom is basic 
        elif acid_or_basic == "B":
            # and if pKa is more than pH, then add index of atom, pKa, and type to stable list to be protonated
            if pKa >= ph:
                stable_data.append( [idx, pKa, "B"] )
                stable_base.append( [idx, pKa, "B"] )
            else:
                unstable_data.append( [idx, pKa, "B"] )
                unstable_base.append( [idx, pKa, "B"] )
    return stable_acid, unstable_acid, stable_base, unstable_base, stable_data, unstable_data

def modify_acid(at):
    '''
    Deprotonates atom
        at: atom of interest that will be ionized
            (type: rdkit object)
        Returns: ionized rdkit molecule 
    '''
    hnum = at.GetNumExplicitHs()
    at.SetFormalCharge(-1)
    at.SetNumExplicitHs(hnum-1)
    return 

def modify_base(at):
    '''
    Protonates atom
        at: atom of interest that will be ionized
            (type: rdkit object)
        Returns: ionized rdkit molecule 
    '''
    hnum = at.GetNumExplicitHs()
    at.SetFormalCharge(1)
    at.SetNumExplicitHs(hnum+1)
    return 

def modify_stable_pka(new_mol, stable_data):
    '''
    Protonate or deprotonate a molecule.
        new_mol: molecule of interest
            (rdkit molecule object)
        stable_data: use a list from get_pKa_data function
            (type: list)
        Returns: a list of the ionization of the molecule 
            [[first_mol,ionized_mol1,pka1],[ionized_mol1,ionized_mol2,pka2]]
    '''
    new_smis = []
    for pka_data in stable_data:
        idx, pka, acid_or_basic = pka_data
        copy_mol = copy.deepcopy(new_mol)
        original_smiles = Chem.MolToSmiles(copy_mol, canonical=True)
        at = new_mol.GetAtomWithIdx(idx)
        try:
            if acid_or_basic == "A":
                # deprotonate atom
                modify_acid(at)
            elif acid_or_basic == "B":
                # protonate atom
                modify_base(at)
            smi = Chem.MolToSmiles(new_mol, canonical=True)
            new_smis.append([original_smiles,smi, pka])
        # if this error occurs molecule most likely invalid, so just skip and continue
        except (OverflowError, ValueError) as e:
            continue
    return new_smis

def modify_unstable_pka(mol, unstable_data, i):
    '''
    If the molecule is an acid and has a pKa more than pH, then it is considered unstable.
    If the molecule is a base and has a pKa less than pH, then it is considered unstable.
    This function is only used in protonate_mol function.
    '''
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
            try:
                if acid_or_basic == "A":
                    modify_acid(at)
                elif acid_or_basic == "B":
                    modify_base(at)
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
                new_unsmis.append(smi)
            except (OverflowError, ValueError) as e:
                continue
        
    return new_unsmis

def ionize_mol(smi, ph):
    print(smi)
    omol = Chem.MolFromSmiles(smi)
    # run pka prediction of molecule; returns base_dict, acid_dict, and smiles object
    obase_dict, oacid_dict, omol = predict(omol)
    # get molecule object with each ionziable atom containing pka and A or B type info
    mc = modify_mol(omol, oacid_dict, obase_dict)

    # separates the modifications between acidic and basic when used in modify_stable_pka()
    amol = deepcopy(mc)
    bmol = deepcopy(mc)

    # separate the prediction based on stability of ionization
    stable_acid, unstable_acid, stable_base, unstable_base, stable_data, unstable_data= get_pKa_data(mc, ph)

    # stable_data follows chem rules, all pKas should be smaller than pH (acid), all pKas should be larger than pH (base)
    # rearrange stable_acid with smallest pKa value being first
    stable_acid.sort(key=lambda stable_acid: stable_acid[1])
    # rearrange unstable_acid with smallest pKa value being first
    # these are unlikely to be deprotonated, but value closest to pH is closest to 50% deprotonated
    unstable_acid.sort(key=lambda unstable_acid: unstable_acid[1])

    # rearramge stable_base with biggest pKa value being first
    stable_base.sort(key=lambda stable_base: stable_base[1], reverse=True)
    unstable_base.sort(key=lambda unstable_base: unstable_base[1], reverse=True)

    stable_asmi, unstable_asmi = [],[]
    stable_bsmi, unstable_bsmi = [],[]
    if len(stable_acid) > 0:
        stable_asmi = modify_stable_pka(amol, stable_acid)
    if len(unstable_acid) > 0:
        unstable_asmi = modify_stable_pka(amol, unstable_acid)
    if len(stable_base) > 0:
        stable_bsmi = modify_stable_pka(bmol, stable_base)
    if len(unstable_base) > 0:
        unstable_bsmi = modify_stable_pka(bmol, unstable_base)
    
    stable_smi = stable_asmi + stable_bsmi
    unstable_smi = unstable_asmi + unstable_bsmi

    return stable_smi, unstable_smi

def protonate_mol(smi, ph):
    '''
    Ionization of all possible sites for a molecule given.
        Returns: 
            new_smis: list of ionized smiles
                (type: list)
    '''
    # make smiles into object to be read by rdkit
    omol = Chem.MolFromSmiles(smi)
    # run pka prediction of molecule; returns base_dict, acid_dict, and smiles object
    obase_dict, oacid_dict, omol = predict(omol)
    # get molecule object with each ionziable atom containing pka and A or B type info
    mc = modify_mol(omol, oacid_dict, obase_dict)
    _,_,_x,_y,stable_data, unstable_data = get_pKa_data(mc, ph)
    new_smis = []
    n = len(unstable_data)
    # if based on pKa and pH rules, the deprotonated or protonated state is viable, then use stable_data only
    if n == 0:
        new_mol = deepcopy(mc)
        modify_stable_pka(new_mol, stable_data)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
        new_smis.append(smi)
    # else use molecules in unstable_data
    else:
        for i in range(n + 1):
            # use molecules in stable_data if available
            new_mol = deepcopy(mc)
            modify_stable_pka(new_mol, stable_data)
            if i == 0:
                new_smis.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))))
            new_unsmis = modify_unstable_pka(new_mol, unstable_data, i)
            new_smis.extend(new_unsmis)
    return new_smis

def load_data(path):
    '''
    Load in any smiles file to have MolGpKa predict pKa and output ionized molecule(s).
        path: path to where the file is located
            (type: str)
        Returns: two lists (stable_smi and unstable_smi)
    '''
    data = pd.read_csv(path, names=['smiles'])
    if data['smiles'].str.contains('acidic:|basic:').any():
        data[['prefix','smiles']] = data['smiles'].str.split(':',expand=True)
    stable_smi, unstable_smi = [], []
    for i in data['smiles']:
        stable, unstable = ionize_mol(i, ph=7.4)
        stable_smi.append(stable)
        unstable_smi.append(unstable)
    return stable_smi, unstable_smi

def save_for_t5chem(stable_smi, unstable_smi, path, stable_only):
    '''
    Organize the output from load_data or ionize_mol in a suitable format for T5Chem-pKa.
        stable_smi: stable molecules containing [base_molecule, ionized molecule, pka]
            (type: list)
        unstable_smi: unstable molecules containing [base_molecule, ionized molecule, pka]
            (type: list)
        path: path to where you want to have the made files in
            (type: str)
        Returns: 4 files (two source files, two target files)
    '''
    micropka_source = open(str(path)+'micropka_train.source', 'w')
    micropka_target = open(str(path)+'micropka_train.target', 'w')
    seq2seq_source = open(str(path)+'seq2seq_train.source', 'w')
    seq2seq_target = open(str(path)+'seq2seq_train.target', 'w')

    if stable_only == False:
        all_data = stable_smi + unstable_smi
    else:
        all_data = stable_only
    data = [item for sublist in all_data for item in sublist if sublist]
    for i in data:
        first_mol, second_mol, pka = i 
        print(first_mol+'>>'+second_mol, file=micropka_source)
        print(pka, file=micropka_target)
        print(first_mol, file=seq2seq_source)
        print(second_mol, file=seq2seq_target)
    micropka_source.close()
    micropka_target.close()
    seq2seq_source.close()
    seq2seq_target.close()

if __name__=="__main__":
    x,y = load_data('/scratch/cii2002/pka/test.source')
    print(x)
    print(y)
    #save_for_t5chem(x,y, '/scratch/cii2002/',stable_only=False)


   

