import argparse
import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import PandasTools
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdf",
        type=str,
        required=True,
        help="path to directory that contains inputs of seq2seq model (.source file)",
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="path to directory that contains inputs of seq2seq model (.source file)",
    )
    args = parser.parse_args()

    sdf = args.sdf
    og = pd.read_csv(args.original)
    # make sdf into DataFrame
    df = PandasTools.LoadSDF(sdf, embedProps=True, molColName=None, smilesName='smiles_ionized_by_epik')
    # keep what is needed
    df = df[['smiles_ionized_by_epik', 'r_epik_pKa_1', 'i_epik_pKa_atom_1', 'r_epik_pKa_2', 'i_epik_pKa_atom_2']]
    df = pd.concat([df, og['BasicOrAcid'],og['acd_pka'], og['smiles']], axis=1)
    # rename columns
    df = df.rename(columns={'r_epik_pKa_1': 'pKa_1','i_epik_pKa_atom_1': 'pKa_atom_1','r_epik_pKa_2':'pKa_2','i_epik_pKa_atom_2':'pKa_atom_2'})

    # figure out Epik pKa is closest to ACD lab's pka
    breakpoint()
    for col in ['pKa_1', 'pKa_2', 'acd_pka']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['diff_1'] = (df['pKa_1'] - df['acd_pka']).abs()
    df['diff_2'] = (df['pKa_2'] - df['acd_pka']).abs()

    # Decide which pKa to keep
    use_1 = df['diff_1'] <= df['diff_2']
    use_1 = use_1 | df['diff_2'].isna()
    use_1 = use_1 & ~df['diff_1'].isna()

    # Select closest pKa and atom
    df['epik_pka'] = np.where(use_1, df['pKa_1'], df['pKa_2'])
    df['Atom'] = np.where(use_1, df['pKa_atom_1'], df['pKa_atom_2'])

    # drop columns
    df = df.drop(columns=['diff_1', 'diff_2','pKa_1','pKa_2','pKa_atom_1','pKa_atom_2'])
    # save csv
    df.to_csv('epik_predicts.csv', index=False)

if __name__ == "__main__":
    main() 