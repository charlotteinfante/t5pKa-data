import argparse
import pandas as pd
import rdkit
from rdkit.Chem import PandasTools


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
    df = df[['smiles_ionized_by_epik', 'r_epik_pKa_1', 'i_epik_pKa_atom_1']]
    df = pd.concat([df, og['BasicOrAcid'],og['acd_pka'], og['smiles']], axis=1)
    # rename columns
    df = df.rename(columns={'r_epik_pKa_1': 'pKa_1','i_epik_pKa_atom_1': 'pKa_atom_1'})
    
    # save csv
    df.to_csv('epik_predicts.csv', index=False)

if __name__ == "__main__":
    main() 