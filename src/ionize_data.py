import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from protonate import modify_mol
from protonate import get_pKa_data
from protonate import modify_stable_pka
from copy import deepcopy
import pdb

def read_data(path):
    data = pd.read_csv(path)
    df_acid = data[data['BasicOrAcid'] == 'A']
    df_basic = data[data['BasicOrAcid'] == 'B']

    df_acid = df_acid[:10]
    df_basic = df_basic[:10]

    acid_list, basic_list = [], []
    breakpoint()
    for smiles in df_acid['smiles'].unique():
        smiles_df = df_acid[df_acid['smiles'] == smiles]
        atom_pka_dict = dict(zip(smiles_df['Atom'], smiles_df['acd_pka']))
        acid_list.append([smiles, atom_pka_dict])
    
    for smiles in df_basic['smiles'].unique():
        smiles_df = df_basic[df_basic['smiles'] == smiles]
        atom_pka_dict = dict(zip(smiles_df['Atom'], smiles_df['acd_pka']))
        basic_list.append([smiles, atom_pka_dict])

    combined_dict = {}
    for smiles, acid_pka_dict in acid_list:
        if smiles not in combined_dict:
            combined_dict[smiles] = [smiles, acid_pka_dict, {}]  # Initialize with acid pKa and empty basic dict
        else:
            combined_dict[smiles][1] = acid_pka_dict  # Update existing entry with acid pKa

    for smiles, basic_pka_dict in basic_list:
        if smiles not in combined_dict:
            combined_dict[smiles] = [smiles, {}, basic_pka_dict]  # Initialize with empty acid dict and basic pKa
        else:
            combined_dict[smiles][2] = basic_pka_dict  # Update existing entry with basic pKa
    together = list(combined_dict.values())
    return together

def ionize(data, ph):
    # list
    breakpoint()
    organized_information = read_data(data)
    for i in organized_information:
        # make smiles into object to be read by rdkit
        omol = Chem.MolFromSmiles(i[0])
        mc = modify_mol(omol, i[1], i[2])
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
        print(stable_smi, unstable_smi)
    return stable_smi, unstable_smi
        


if __name__=="__main__":
    ionize('/vast/cii2002/full_ACD_CHEMBL_pka.csv', 7.4)
