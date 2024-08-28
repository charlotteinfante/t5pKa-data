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

def read_data(path):
    data = pd.read_csv(path)
    molecule_pka_dict = {}
    for index, row in data.iterrows():
        smiles = row['smiles']
        category = row['BasicOrAcid']
        pka_value = row['acd_pka']
        if smiles not in molecule_pka_dict:
            molecule_pka_dict[smiles] = {'A':[],'B':[]}
        molecule_pka_dict[smiles][category].append(pka_value)
