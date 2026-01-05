import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from protonate import get_pKa_data
from protonate import modify_stable_pka
from protonate import save_for_t5chem
from copy import deepcopy
import pdb

def read_data(path):
    data = pd.read_csv(path)
    # separate dataframe based on whether acidic or basic 
    #breakpoint()
    data["Atom"] = pd.to_numeric(data["Atom"], errors="coerce") - 1
    df_acid = data[data['BasicOrAcid'] == 'A']
    df_basic = data[data['BasicOrAcid'] == 'B']

    acid_list, basic_list = [], []
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
    return mol

def ionize(data, ph):
    # list
    organized_information = read_data(data)
    stable_acid_smi, stable_basic_smi = [], []
    unstable_acid_smi, unstable_basic_smi = [],[]
    for i in organized_information:
        # make smiles into object to be read by rdkit
        if not Chem.MolFromSmiles(i[0]):
            continue
        omol = Chem.AddHs(Chem.MolFromSmiles(i[0]))
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
            stable_asmi = modify_stable_pka(AllChem.RemoveHs(amol), stable_acid)
            stable_acid_smi.append(stable_asmi)
        if len(unstable_acid) > 0:
            unstable_asmi = modify_stable_pka(AllChem.RemoveHs(amol), unstable_acid)
            unstable_acid_smi.append(unstable_asmi)
        if len(stable_base) > 0:
            stable_bsmi = modify_stable_pka(AllChem.RemoveHs(bmol), stable_base)
            stable_basic_smi.append(stable_bsmi)
        if len(unstable_base) > 0:
            unstable_bsmi = modify_stable_pka(AllChem.RemoveHs(bmol), unstable_base)
            unstable_basic_smi.append(unstable_bsmi)
    stable_smi = stable_acid_smi + stable_basic_smi
    unstable_smi = unstable_acid_smi + unstable_basic_smi
    print(stable_smi, unstable_smi)
    save_for_t5chem(stable_smi, unstable_smi, '/scratch/cii2002/', stable_only=False)
    return stable_smi, unstable_smi
        

if __name__=="__main__":
    ionize('/scratch/cii2002/MolGpKa-data/src/epik/epik_test.csv', 7.4)
    #ionize('/vast/cii2002/full_ACD_CHEMBL_pka.csv', 7.4)