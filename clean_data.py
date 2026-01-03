import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


data = pd.read_table('/scratch/cii2002/pka/chembl_version_25/chembl_25_acd_pka.tsv')

# make sure smiles are represented as rdkit canonical smiles and look for any invalid molecules
canonical_smiles, error_index = [], []
for x,i in enumerate(data['canonical_smiles']):
    try:
        mol = Chem.MolFromSmiles(i)
        canonical_smiles.append(Chem.MolToSmiles(mol), canonical=True)
        if mol is None:
            error_index.append(x)
    except Exception as e:
        continue
print(canonical_smiles)
# drop invalid molecules
data = data.drop(error_index)
data['rdkit_canonical_smiles'] = canonical_smiles

# remove salts from molecules
remover = SaltRemover() # use default saltremover
clean_mol = []
for i in data['rdkit_canonical_smiles']:
    stripped = remover.StripMol(Chem.MolFromSmiles(i))
    clean_mol.append(stripped)

def remove_salts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Get the fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    # Keep the largest fragment
    largest_frag = max(frags, key=lambda frag: frag.GetNumAtoms())
    # Convert back to SMILES
    clean_smiles = Chem.MolToSmiles(largest_frag, canonical=True)
    return clean_smiles

    df_with_period['Cleaned_SMILES'] = df_with_period['smiles'].apply(remove_salts)