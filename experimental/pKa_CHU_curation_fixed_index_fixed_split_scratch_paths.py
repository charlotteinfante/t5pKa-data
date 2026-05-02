import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import rdkit
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize as std
from rdkit.Chem import inchi
import ast
import os

USE_CHIRALITY = False

def canonicalize(smiles):
    '''
    Canonicalize SMILES using RDKit
    '''
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def mol_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    m2 = rdMolStandardize.Cleanup(mol)
    return Chem.MolToInchiKey(m2)

def canonicalize_micropka_pair(pair_str: str) -> str:

    if pd.isna(pair_str):
        return None

    try:
        smi1, smi2 = str(pair_str).split(">>", 1)

        m1 = Chem.MolFromSmiles(smi1.strip())
        m2 = Chem.MolFromSmiles(smi2.strip())

        if m1 is None or m2 is None:
            return None

        cs1 = Chem.MolToSmiles(
                m1,
                canonical=True,
                isomericSmiles=True
            )

        cs2 = Chem.MolToSmiles(
                m2,
                canonical=True,
                isomericSmiles=True
            )

        return f"{cs1}>>{cs2}"

    except Exception:
        return None
    # split into the two SMILES: protonated molecule and deprotonated molecule
    #try:
    #    smi1, smi2 = pair_str.split('>>', 1)
    #except ValueError:
    #    # if it doesn't have “>>”, just return it unchanged
    #    return pair_str
    
    # parse with RDKit
    #m1 = Chem.MolFromSmiles(smi1)
    #m2 = Chem.MolFromSmiles(smi2)
    
    # canonicalize (fall back to empty if parse fails)
    #cs1 = Chem.MolToSmiles(m1, canonical=True) if m1 else ''
    #cs2 = Chem.MolToSmiles(m2, canonical=True) if m2 else ''
    #return f"{cs1}>>{cs2}"


def mol_and_fp(df, smiles_column, radius=2, nBits=2048):
    '''
    Get the fingerprints of molecules from smiles in a dataframe 
    '''
    df = df.copy()
    df['Mol'] = df[smiles_column].apply(Chem.MolFromSmiles)
    # drop any non-molecule
    df = df[df['Mol'].notna()]  
    df['FP'] = df['Mol'].apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits))
    df = df.drop(['Mol'], axis=1)
    return df

def find_micropka_tanimoto_overlaps(df1, df2, threshold: float = 0.85, radius: int = 2, nBits: int = 2048):
    """
    Compare every row in df1 vs df2, splitting 'micropka input' on '>>'
    into prot_smiles / deprot_smiles. Returns a list of matches where
    either prot and deprot fingerprint similarity ≥ threshold.
    Each match dict tells you which half (prot/deprot) passed the cutoff.
    """
    # 1) Split & fingerprint both tables (in‑place)
    for df in (df1, df2):
        if 'prot_smiles' not in df:
            df[['prot_smiles','deprot_smiles']] = (
                df['micropka input']
                  .str.split('>>', expand=True)
            )
            df['FP_prot']   = df['prot_smiles'].apply(
                lambda sm: AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(sm), radius, nBits
                )
            )
            df['FP_deprot'] = df['deprot_smiles'].apply(
                lambda sm: AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(sm), radius, nBits
                )
            )

    # 2) Cache the df2 fingerprints & smiles
    fps2_p   = df2['FP_prot'].tolist()
    fps2_d   = df2['FP_deprot'].tolist()
    pka2     = df2['target'].tolist()
    prot2    = df2['prot_smiles'].tolist()
    deprot2  = df2['deprot_smiles'].tolist()
    cit2     = df2['citation'].tolist()
    raw2     = df2['micropka input'].tolist()

    matches = []
    # 3) Loop through df1
    for i, r1 in df1.iterrows():
        sims_p = DataStructs.BulkTanimotoSimilarity(r1['FP_prot'], fps2_p)
        sims_d = DataStructs.BulkTanimotoSimilarity(r1['FP_deprot'], fps2_d)

        for j, (sp, sd) in enumerate(zip(sims_p, sims_d)):
            # did either half clear the bar?
            if sp >= threshold and sd >= threshold:
                # pick the winning half + score
                if sp >= sd:
                    half, score = 'prot', sp
                else:
                    half, score = 'deprot', sd

                matches.append({
                    'idx1':            i,
                    'idx2':            j,
                    'half':            half,
                    'prefix':          r1['prefix'],
                    'micropka_input_1':      r1['micropka input'],
                    'micropka_input_2':      raw2[j],
                    'FP_prot_1':       r1['FP_prot'],
                    'FP_deprot_1':     r1['FP_deprot'],
                    'FP_prot_2':       fps2_p[j],
                    'FP_deprot_2':     fps2_d[j],
                    'pKa_1':           r1['target'],
                    'pKa_2':           pka2[j],
                    'citation_1':      r1['citation'],
                    'citation_2':      cit2[j],
                    'Tanimoto_prot':   sp,
                    'Tanimoto_deprot': sd,
                    'Tanimoto':        score,
                })

    return matches

def find_tanimoto_matches(df1, df2, threshold=1.00):
    """
    Return all Tanimoto≥threshold hits between df1 vs df2 *with the same* prefix.
    Each hit dict still carries 'prefix' so you know acidic/basic.
    """
    matches = []
    FP2       = df2['FP'].tolist()
    prefix2   = df2['prefix'].tolist()
    smiles2   = df2['canonical smiles'].tolist()
    micropka2 = df2['micropka input'].tolist()
    pka2      = df2['target'].tolist()
    cit2      = df2['citation'].tolist()

    for _, r1 in df1.iterrows():
        sims = DataStructs.BulkTanimotoSimilarity(r1['FP'], FP2)
        for j, sim in enumerate(sims):
            if sim >= threshold and r1['prefix'] == prefix2[j]:
                matches.append({
                    'prefix':           r1['prefix'],
                    'SMILES_1':         r1['canonical smiles'],
                    'micropka_input_1': r1['micropka input'],
                    'pKa_1':            r1['target'],
                    'citation_1':       r1['citation'],
                    'FP_1':             r1['FP'],
                    'SMILES_2':         smiles2[j],
                    'micropka_input_2': micropka2[j],
                    'pKa_2':            pka2[j],
                    'citation_2':       cit2[j],
                    'FP_2':             FP2[j],
                    'Tanimoto':         sim
                })
    return matches

def classify_reaction(reaction_smiles):
    '''
    Give a dataset the prefix "acidic" for pKa or "basic" for pKaH if it is missing this assignment due to other datasets including it. 
    The overall model will not be trained with these prefixes. 
    '''
    try:
        reactant_smiles, product_smiles = reaction_smiles.split(">>")
        mol1 = Chem.MolFromSmiles(reactant_smiles)
        mol2 = Chem.MolFromSmiles(product_smiles)

        if mol1 is None or mol2 is None:
            return None

        atoms1 = mol1.GetAtoms()
        atoms2 = mol2.GetAtoms()

        # Create rough fingerprint per atom: (symbol, number of neighbors, charge)
        def atom_signature(atom):
            return (atom.GetSymbol(), len(atom.GetNeighbors()), atom.GetFormalCharge())

        sigs1 = [atom_signature(a) for a in atoms1]
        sigs2 = [atom_signature(a) for a in atoms2]

        # Try to match atoms by signature (symbol + neighborhood) and look at charge delta
        for i1, sig1 in enumerate(sigs1):
            for i2, sig2 in enumerate(sigs2):
                if sig1[:2] == sig2[:2]:  # same atom symbol and connectivity
                    charge_diff = sig2[2] - sig1[2]
                    if charge_diff == -1:
                        # atom lost a proton: charge decreased
                        if sig1[2] == +1:
                            return "basic"
                        elif sig1[2] == 0:
                            return "acidic"
        return None
    except:
        return None

def compare_micropka_inputs(df):
    '''
    Compare the ionization centers between D2A-pKa and our data.
        df: DataFrame that can needs to have gone through find_taniomto_similarty function. 
            (type: pandas dataframe)
        Returns: 
            pandas dataframe with all molecules that have a different ionization center between our data and another other dataset. 
    '''
    mismatches = []
    for i, row in df.iterrows():
        try: 
            prot_smiles_1 = canonicalize(row['micropka_input_1'].split('>>')[0])
            deprot_mol_1 = Chem.MolFromSmiles(canonicalize(row['micropka_input_1'].split('>>')[1]))
            atom_idx_of_deprot_1 = [(atom.GetIdx(), atom.GetSymbol(), atom.GetFormalCharge()) for atom in deprot_mol_1.GetAtoms() if atom.GetFormalCharge() != 0]
            prot_smiles_2 = canonicalize(row['micropka_input_2'].split('>>')[0])
            deprot_mol_2 = Chem.MolFromSmiles(canonicalize(row['micropka_input_2'].split('>>')[1]))
            atom_idx_of_deprot_2 = [(atom.GetIdx(), atom.GetSymbol(), atom.GetFormalCharge()) for atom in deprot_mol_2.GetAtoms() if atom.GetFormalCharge() != 0]

            #if set(atom_idx_of_deprot_1) != set(atom_idx_of_deprot_2): 
            if set(atom_idx_of_deprot_1) != set(atom_idx_of_deprot_2) and prot_smiles_1 == prot_smiles_2:
                mismatches.append(i)
        except Exception as e:
            print('Error processing row' + str(i) + ' : ' + str(e))
    return df.loc[mismatches].copy()

def random_split(data, train_ratio, test_ratio, seed):
    '''
    Performs random splitting based on a given ratio for the training, validation, and test sets. 
        data: file that contains the dataset
            (type: pandas dataframe or csv)
        train_ratio: size of the training set
            (type: float)
        test_ratio : size of the test set 
            (type: float)
        seed: the seed you want each run to hold
            (type: int)
        Returns: 3 pandas dataframe 
        
        example for an 8:1:1 splitting:
            train, val, test = random_split(dataset, 0.8, 0.1, 42)
        *This is a slightly modified version of sklearn's train_test_split function*
    '''
    np.random.seed(seed)
    shuffle_data = np.random.permutation(len(data))
    train_indices = shuffle_data[:int(len(data)*train_ratio)]
    val_indices = shuffle_data[int(len(data)*train_ratio):int(len(data)*(1.0-test_ratio))]
    test_indices = shuffle_data[int(len(data)*(1.0-test_ratio)):]
    return data.iloc[train_indices], data.iloc[val_indices], data.iloc[test_indices]


def total_abs_charge(mol):
    return sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())

def count_neg_oxygens(mol):
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O' and a.GetFormalCharge() == -1)

def choose_pre_and_post(left_smiles, right_smiles):
    ml = Chem.MolFromSmiles(left_smiles)
    mr = Chem.MolFromSmiles(right_smiles)
    if not ml or not mr:
        return None, None

    # Heuristic 1: smaller total |charge| → pre-ionization
    tal, tar = total_abs_charge(ml), total_abs_charge(mr)
    if tal != tar:
        pre, post = (ml, mr) if tal < tar else (mr, ml)
    else:
        # Heuristic 2: fewer negatively charged oxygens → pre
        nl, nr = count_neg_oxygens(ml), count_neg_oxygens(mr)
        if nl != nr:
            pre, post = (ml, mr) if nl < nr else (mr, ml)
        else:
            # Heuristic 3: more hydrogens → pre
            hl, hr = sum(a.GetTotalNumHs() for a in ml.GetAtoms()), sum(a.GetTotalNumHs() for a in mr.GetAtoms())
            pre, post = (ml, mr) if hl >= hr else (mr, ml)

    return (Chem.MolToSmiles(pre, canonical=True),
            Chem.MolToSmiles(post, canonical=True))

def seq2seq_pair(pair):
    left, right = pair.split('>>', 1)
    return choose_pre_and_post(left.strip(), right.strip())

_lfc = std.LargestFragmentChooser()
_uncharger = std.Uncharger()
def _to_mol(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = std.Cleanup(m)                # normalize, reionize (std cleanup)
        m = _lfc.choose(m)                # take largest fragment
        m = _uncharger.uncharge(m)        # remove formal charges where possible
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def charge_invariant_key(smi: str) -> str:
    """
    Returns a canonical SMILES key that is insensitive to +/-1 charge differences.
    """
    m = _to_mol(smi)
    if m is None:
        return f"BAD:{smi}"   # keep a stable key even if parsing fails
    # isomeric=True keeps stereo; set to False if you want to merge enantiomers too
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def group_random_split(df, group_col="group_id", train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    assert np.isclose(train_frac + val_frac + test_frac, 1.0), "Fractions must sum to 1."

    # accept either a column name (str) or an array-like of group labels
    if isinstance(group_col, str):
        groups = df[group_col].values
    else:
        groups = np.asarray(group_col)
        assert len(groups) == len(df), "If group_col is not a string, it must be same length as df."

    idx = np.arange(len(df))

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = groups[temp_idx]  # use the same group vector, not temp_df[group_col]
    temp_idx2 = np.arange(len(temp_df))

    val_size_within_temp = val_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size_within_temp, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_idx2, groups=temp_groups))

    train = df.iloc[train_idx].copy()
    val   = temp_df.iloc[val_rel].copy()
    test  = temp_df.iloc[test_rel].copy()

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def take_groups(src_df, allowed_groups, n_groups=None):
    """
    Take all rows whose group_id is in allowed_groups (optionally limited to the first n_groups).
    Returns (taken_df, remaining_df).
    """
    if n_groups is not None:
        allowed_groups = set(list(allowed_groups)[:n_groups])
    mask = src_df['group_id'].isin(allowed_groups)
    return src_df[mask].copy(), src_df[~mask].copy()

def pick_n_groups(src_df, n_groups):
    """
    Deterministically pick the first n unseen groups from src_df (by appearance order).
    """
    seen = set()
    picked = []
    for g in src_df['group_id']:
        if g not in seen:
            seen.add(g)
            picked.append(g)
            if len(picked) >= n_groups:
                break
    return picked


# amino acids from Vanderbilt University
aa = [{'prefix': 'acidic','micropka input': 'NC(N)=[N+]CCCC([NH3+])C(=O)O>>NC(N)=[N+]CCCC([NH3+])C(=O)[O-]','target': 1.82,'Name': 'Arg'},
 {'prefix': 'basic','micropka input': 'NC(N)=[N+]CCCC([NH3+])C(=O)[O-]>>NC(N)=[N+]CCCC(N)C(=O)[O-]','target': 8.99,'Name': 'Arg'},
 {'prefix': 'basic','micropka input': 'NC(N)=[N+]CCCC(N)C(=O)[O-]>>NC(N)=NCCCC(N)C(=O)[O-]','target': 12.48,'Name': 'Arg'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CS)C(=O)O>>[NH3+]C(CS)C(=O)[O-]','target': 1.92,'Name': 'Cys'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CS)C(=O)[O-]>>[NH3+]C(C[S-])C(=O)[O-]','target': 8.37,'Name': 'Cys'},
 {'prefix': 'basic','micropka input': '[NH3+]C(C[S-])C(=O)[O-]>>NC(C[S-])C(=O)[O-]','target': 10.7,'Name': 'Cys'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CCC(=O)O)C(=O)O>>[NH3+]C(CCC(=O)O)C(=O)[O-]','target': 2.1,'Name': 'Glu'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CCC(=O)O)C(=O)[O-]>>[NH3+]C(CCC(=O)[O-])C(=O)[O-]','target': 4.07,'Name': 'Glu'},
 {'prefix': 'basic','micropka input': '[NH3+]C(CCC(=O)[O-])C(=O)[O-]>>NC(CCC(=O)[O-])C(=O)[O-]','target': 9.47,'Name': 'Glu'},
 {'prefix': 'acidic','micropka input': '[NH3+]CCCCC([NH3+])C(=O)O>>[NH3+]CCCCC([NH3+])C(=O)[O-]','target': 2.16,'Name': 'Lys'},
 {'prefix': 'basic','micropka input': '[NH3+]CCCCC([NH3+])C(=O)[O-]>>NC(CCCC[NH3+])C(=O)[O-]','target': 9.06,'Name': 'Lys'},
 {'prefix': 'basic','micropka input': 'NC(CCCC[NH3+])C(=O)[O-]>>NCCCCC(N)C(=O)[O-]','target': 10.54,'Name': 'Lys'},
 {'prefix': 'acidic','micropka input': 'NC(=O)CCC([NH3+])C(=O)O>>NC(=O)CCC([NH3+])C(=O)[O-]','target': 2.17,'Name': 'Gln'},
 {'prefix': 'basic','micropka input': 'NC(=O)CCC([NH3+])C(=O)[O-]>>NC(=O)CCC(N)C(=O)[O-]','target': 9.13,'Name': 'Gln'},
 {'prefix': 'acidic','micropka input': 'CC([NH3+])C(=O)O>>CC([NH3+])C(=O)[O-]','target': 2.35,'Name': 'Ala'},
 {'prefix': 'basic','micropka input': 'CC([NH3+])C(=O)[O-]>>CC(N)C(=O)[O-]','target': 9.87,'Name': 'Ala'},
 {'prefix': 'acidic','micropka input': 'NC(=O)CC([NH3+])C(=O)O>>NC(=O)CC([NH3+])C(=O)[O-]','target': 2.14,'Name': 'Asn'},
 {'prefix': 'basic','micropka input': 'NC(=O)CC([NH3+])C(=O)[O-]>>NC(=O)CC(N)C(=O)[O-]','target': 8.72,'Name': 'Asn'},
 {'prefix': 'acidic','micropka input': '[NH3+]CC(=O)O>>[NH3+]CC(=O)[O-]','target': 2.35,'Name': 'Gly'},
 {'prefix': 'basic','micropka input': '[NH3+]CC(=O)[O-]>>NCC(=O)[O-]','target': 9.78,'Name': 'Gly'},
 {'prefix': 'acidic','micropka input': 'CCC(C)C([NH3+])C(=O)O>>CCC(C)C([NH3+])C(=O)[O-]','target': 2.32,'Name': 'Ile'},
 {'prefix': 'basic','micropka input': 'CCC(C)C([NH3+])C(=O)[O-]>>CCC(C)C(N)C(=O)[O-]','target': 9.76,'Name': 'Ile'},
 {'prefix': 'acidic','micropka input': 'CC(C)CC([NH3+])C(=O)O>>CC(C)CC([NH3+])C(=O)[O-]','target': 2.33,'Name': 'Leu'},
 {'prefix': 'basic','micropka input': 'CC(C)CC([NH3+])C(=O)[O-]>>CC(C)CC(N)C(=O)[O-]','target': 9.74,'Name': 'Leu'},
 {'prefix': 'acidic','micropka input': 'CSCCC([NH3+])C(=O)O>>CSCCC([NH3+])C(=O)[O-]','target': 2.13,'Name': 'Met'},
 {'prefix': 'basic','micropka input': 'CSCCC([NH3+])C(=O)[O-]>>CSCCC(N)C(=O)[O-]','target': 9.28,'Name': 'Met'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CC(=O)O)C(=O)O>>[NH3+]C(CC(=O)O)C(=O)[O-]','target': 1.99,'Name': 'Asp'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CC(=O)O)C(=O)[O-]>>[NH3+]C(CC(=O)[O-])C(=O)[O-]','target': 3.9,'Name': 'Asp'},
 {'prefix': 'basic','micropka input': '[NH3+]C(CC(=O)[O-])C(=O)[O-]>>NC(CC(=O)[O-])C(=O)[O-]','target': 9.9,'Name': 'Asp'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CC1=CN=C[NH+]1)C(=O)O>>[NH3+]C(CC1=CN=C[NH+]1)C(=O)[O-]','target': 1.8,'Name': 'His'},
 {'prefix': 'basic','micropka input': '[NH3+]C(CC1=CN=C[NH+]1)C(=O)[O-]>>[NH3+]C(Cc1cnc[nH]1)C(=O)[O-]','target': 6.04,'Name': 'His'},
 {'prefix': 'basic','micropka input': '[NH3+]C(Cc1cnc[nH]1)C(=O)[O-]>>NC(Cc1cnc[nH]1)C(=O)[O-]','target': 9.33,'Name': 'His'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(Cc1ccc(O)cc1)C(=O)O>>[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]','target': 2.2,'Name': 'Tyr'},
 {'prefix': 'basic','micropka input': '[NH3+]C(Cc1ccc(O)cc1)C(=O)[O-]>>NC(Cc1ccc(O)cc1)C(=O)[O-]','target': 9.21,'Name': 'Tyr'},
 {'prefix': 'acidic','micropka input': 'NC(Cc1ccc(O)cc1)C(=O)[O-]>>NC(Cc1ccc([O-])cc1)C(=O)[O-]','target': 10.46,'Name': 'Tyr'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(Cc1ccccc1)C(=O)O>>[NH3+]C(Cc1ccccc1)C(=O)[O-]','target': 2.2,'Name': 'Phe'},
 {'prefix': 'basic','micropka input': '[NH3+]C(Cc1ccccc1)C(=O)[O-]>>NC(Cc1ccccc1)C(=O)[O-]','target': 9.31,'Name': 'Phe'},
 {'prefix': 'acidic','micropka input': 'O=C(O)C1CCC[NH2+]1>>O=C([O-])C1CCC[NH2+]1','target': 1.95,'Name': 'Pro'},
 {'prefix': 'basic','micropka input': 'O=C([O-])C1CCC[NH2+]1>>O=C([O-])C1CCCN1','target': 10.64,'Name': 'Pro'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(CO)C(=O)O>>[NH3+]C(CO)C(=O)[O-]','target': 2.19,'Name': 'Ser'},
 {'prefix': 'basic','micropka input': '[NH3+]C(CO)C(=O)[O-]>>NC(CO)C(=O)[O-]','target': 9.21,'Name': 'Ser'},
 {'prefix': 'acidic','micropka input': 'CC(O)C([NH3+])C(=O)O>>CC(O)C([NH3+])C(=O)[O-]','target': 2.09,'Name': 'Thr'},
 {'prefix': 'basic','micropka input': 'CC(O)C([NH3+])C(=O)[O-]>>CC(O)C(N)C(=O)[O-]','target': 9.1,'Name': 'Thr'},
 {'prefix': 'acidic','micropka input': '[NH3+]C(Cc1c[nH]c2ccccc12)C(=O)O>>[NH3+]C(Cc1c[nH]c2ccccc12)C(=O)[O-]','target': 2.46,'Name': 'Trp'},
 {'prefix': 'basic','micropka input': '[NH3+]C(Cc1c[nH]c2ccccc12)C(=O)[O-]>>NC(Cc1c[nH]c2ccccc12)C(=O)[O-]','target': 9.41,'Name': 'Trp'},
 {'prefix': 'acidic','micropka input': 'CC(C)C([NH3+])C(=O)O>>CC(C)C([NH3+])C(=O)[O-]','target': 2.29,'Name': 'Val'},
 {'prefix': 'basic','micropka input': 'CC(C)C([NH3+])C(=O)[O-]>>CC(C)C(N)C(=O)[O-]','target': 9.74,'Name': 'Val'}]

aa_df = pd.DataFrame(aa)
aa_df['citation'] = ['Vanderbilt University' for i in range(len(aa_df))]
aa_df[['smiles','deprotonated smiles']] = aa_df['micropka input'].str.split('>>',expand=True)
smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in aa_df['smiles']]
aa_df['canonical smiles'] = smiles_list
aa_df = mol_and_fp(aa_df, 'canonical smiles')

# import datasets
print('======================================= IMPORTING DATASETS: LENGTH OF DATASETS SHOWN BELOW ================================================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
ml_meets_pka = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/ml_meets_pka.csv')
ml_meets_pka = ml_meets_pka.rename(columns={'marvin_pKa_type': 'prefix'})
ml_meets_pka = ml_meets_pka.rename(columns={'pKa': 'target'})
ml_meets_pka['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in ml_meets_pka['smiles']]
ml_meets_pka['micropka input'] = (ml_meets_pka['micropka input'].astype(str).apply(canonicalize_micropka_pair))
ml_meets_pka['citation'] = ['Machine Learning meets pKa' for i in range(len(ml_meets_pka))]
ml_meets_pka = mol_and_fp(ml_meets_pka, 'canonical smiles')
print('# of molecules in Machines Learning meets pKa: ' + str(len(ml_meets_pka)))

comp_9_ = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/comparing_9_programs_for_pka_prediction_oct_20_2025.csv')
comp_9 = comp_9_.rename(columns={'rdkit canonical SMILES':'smiles','marvin type': 'prefix','pKa':'target','marvin SMILES':'ionized smiles'})
comp_9['micropka input'] = (comp_9['micropka input'].astype(str).apply(canonicalize_micropka_pair))
comp_9['citation'] = ['Comparison of 9 programs for pKa prediction' for i in range(len(comp_9))]
comp_9['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in comp_9['smiles']]
# Comparison of 9 programs has molecules with more than one pKa (0 --> -1) or (+1 --> 0), so for now we will just get rid of them to stay with monoprotic molecules 
comp_9_acidic = comp_9[comp_9['prefix'] == 'acidic']
comp_9_basic = comp_9[comp_9['prefix'] == 'basic']
name_counts_acidic = comp_9_acidic['Name'].value_counts()
name_counts_basic = comp_9_basic['Name'].value_counts()
names_to_drop_acidic = name_counts_acidic[name_counts_acidic > 1].index
names_to_drop_basic = name_counts_basic[name_counts_basic > 1].index
comp_9_filtered_acidic = comp_9[~comp_9['Name'].isin(names_to_drop_acidic)].copy()
comp_9 = comp_9_filtered_acidic[~comp_9_filtered_acidic['Name'].isin(names_to_drop_basic)].copy()
# drop duplicates within the added datasets based on smiles and whether it is acid or base 
# this wil inevitably get rid of any polyprotic molecule with more than 2 pKa values or more than one pKa or pKaH value 
duplicates_comp9 = comp_9[comp_9.duplicated(subset=['canonical smiles','prefix'], keep=False)]
#comp_9 = comp_9.drop_duplicates(subset=['canonical smiles','prefix'], keep=False)
comp_9 = mol_and_fp(comp_9, 'canonical smiles')
print('# of molecules in Comparsion of 9 programs for pKa prediction BEFORE|AFTER removing more than 1 pKa or pKaH per molecule: ' + str(len(comp_9_)) + '|' + str(len(comp_9)))

az_ = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/az_results.csv')
az = az_.rename(columns={'SMILES':'smiles','Marvin type':'prefix','pKa experimental':'target','Marvin SMILES':'ionized smiles'})
az['micropka input'] = (az['micropka input'].astype(str).apply(canonicalize_micropka_pair))
az['citation']  = ['AZ' for i in range(len(az))]
az['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in az['smiles']]
# drop duplicates within the added datasets based on smiles and whether it is acid or base 
# this wil inevitably get rid of any polyprotic molecule with more than 2 pKa values or more than one pKa or pKaH value 
duplicates_az = az[az.duplicated(subset=['canonical smiles','prefix'], keep=False)]
#az = az.drop_duplicates(subset = ['canonical smiles','prefix'], keep=False)
az = mol_and_fp(az, 'canonical smiles')
print('# of molecules in AZ BEFORE|AFTER removing more than 1 pKa or pKaH per molecule: ' + str(len(az_)) + '|' + str(len(az)))

manchester_ = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/manchester_canonical_oct_20_2025.csv')
manchester = manchester_.rename(columns={'rdkit canonical smiles':'smiles','pKa experimental':'target','Marvin Type':'prefix','Marvin SMILES':'ionized smiles'})
manchester['micropka input'] = (manchester['micropka input'].astype(str).apply(canonicalize_micropka_pair))
manchester['citation'] = ['Manchester' for i in range(len(manchester))]
manchester['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in manchester['smiles']]
# drop duplicates within the added datasets based on smiles and whether it is acid or base 
# this wil inevitably get rid of any polyprotic molecule with more than 2 pKa values or more than one pKa or pKaH value 
duplicates_manchester = manchester[manchester.duplicated(subset=['micropka input','prefix'], keep=False)]
#manchester = manchester.drop_duplicates(subset=['micropka input','prefix'], keep=False)
manchester = mol_and_fp(manchester, 'canonical smiles')
print('# of molecules in Manchester BEFORE|AFTER removing more than 1 pKa or pKaH per molecule: ' + str(len(manchester_)) + '|' + str(len(manchester)))

vertex_ = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/vertex_pka_oct_20_2025.csv')
vertex = vertex_.rename(columns={'rdkit canonical smiles':'smiles','pKa experimental':'target','Marvin Type':'prefix','Marvin SMILES':'ionized smiles'})
vertex['micropka input'] = (vertex['micropka input'].astype(str).apply(canonicalize_micropka_pair))
vertex['citation'] = ['Vertex' for i in range(len(vertex))]
vertex['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in vertex['smiles']]
# drop duplicates within the added datasets based on smiles and whether it is acid or base 
# this wil inevitably get rid of any polyprotic molecule with more than 2 pKa values or more than one pKa or pKaH value 
duplicates_vertex = vertex[vertex.duplicated(subset=['canonical smiles','prefix'], keep=False)]
#vertex = vertex.drop_duplicates(subset=['smiles','prefix'], keep=False)
vertex = mol_and_fp(vertex, 'canonical smiles')
print('# of molecules in Vertex BEFORE|AFTER removing more than 1 pKa or pKaH per molecule: ' + str(len(vertex_)) + '|' + str(len(vertex)))

morgan_ = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/morgen_pka.csv')
morgan = morgan_.rename(columns={'rdkit canonical smiles':'smiles','pKa experimental':'target','Marvin Type':'prefix','Marvin SMILES':'ionized smiles'})
morgan['micropka input'] = (morgan['micropka input'].astype(str).apply(canonicalize_micropka_pair))
morgan['citation'] = ['MorgenThaler' for i in range(len(morgan))]
morgan['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in morgan['smiles']]
# drop duplicates within the added datasets based on smiles and whether it is acid or base 
# this wil inevitably get rid of any polyprotic molecule with more than 2 pKa values or more than one pKa or pKaH value 
duplicates_morgan = morgan[morgan.duplicated(subset=['micropka input'], keep=False)]
#morgan = morgan.drop_duplicates(subset=['micropka input'], keep=False)
morgan = mol_and_fp(morgan, 'canonical smiles')
print('# of molecules in Morgan BEFORE|AFTER removing more than 1 pKa or pKaH per molecule: ' + str(len(morgan_)) + '|' + str(len(morgan)))

comp_ab_ = pd.read_csv('/scratch/cii2002/pka/NEW_TRAINING_DATA/DATA_JUNE_24_2025/data/pubchem_molecules_123_oct_20_2025.csv')
comp_ab_['Activity Comment'] = comp_ab_['Activity Comment'].replace({'Acidic pKa': 'acidic','Basic pKa' : 'basic'})
comp_ab = comp_ab_.rename(columns={'SMILES':'smiles','pKa experimental':'target','Marvin Ionized SMILES':'ionized smiles','Activity Comment':'prefix'})
comp_ab['micropka input'] = (comp_ab['micropka input'].astype(str).apply(canonicalize_micropka_pair))
comp_ab['citation'] = ['Comparison of Acidic and Basic pKa Settimo et al.' for i in range(len(comp_ab))]
comp_ab['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in comp_ab['smiles']]
# drop duplicates within the added datasets based on smiles and whether it is acid or base 
# this wil inevitably get rid of any polyprotic molecule with more than 2 pKa values or more than one pKa or pKaH value 
duplicates_comp_ab = comp_ab[comp_ab.duplicated(subset=['canonical smiles','prefix'], keep=False)]
#comp_ab = comp_ab.drop_duplicates(subset=['smiles','prefix'], keep=False)
comp_ab = mol_and_fp(comp_ab, 'canonical smiles')
print('# of molecules in Comparison of Acids and Bases BEFORE|AFTER removing more than 1 pKa or pKaH per molecule: ' + str(len(comp_ab_)) + '|' + str(len(comp_ab)))

collect_all_pKa_pKaH = pd.concat([duplicates_az, duplicates_comp9, duplicates_comp_ab, duplicates_vertex, duplicates_morgan, duplicates_manchester], axis=0)

# import iupac but do not add it to the rest of the datasets; will not be for training 
iupac = pd.read_csv('/scratch/cii2002/pka/iupac/macropka_with_monoprotic_molecules_only.csv')
iupac[['prefix','smiles']] = iupac['prefix_smiles'].str.split(':',expand=True)
iupac['citation']= ['IUPAC' for i in range(len(iupac))]
iupac = mol_and_fp(iupac, 'smiles')
iupac['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in iupac['smiles']]

# import D2A-pKa data
d2a = pd.read_csv('/scratch/cii2002/solvation_free_energies_data/D2A-pKa.csv')
#get only the pka extracted using water solvent 
d2a_oxygen = d2a[d2a['solvent_smiles'] == 'O'].copy()
# apply acidic or basic prefix to go with other datasets; will be removed later 
d2a_oxygen['prefix'] = d2a_oxygen['reaction_smiles'].apply(classify_reaction)
d2a_oxygen[['protonated','deprotonated']] = d2a_oxygen['reaction_smiles'].str.split('>>',expand=True)
d2a_oxygen = mol_and_fp(d2a_oxygen, 'protonated')
d2a_oxygen['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in d2a_oxygen['protonated']]
d2a_oxygen_ = d2a_oxygen.rename(columns={'refs': 'citation','pKa_avg':'target','reaction_smiles':'micropka input'})
d2a_oxygen_['micropka input'] = (d2a_oxygen_['micropka input'].astype(str).apply(canonicalize_micropka_pair))
# fill in D2A-pKa rows whose original `refs` field is an empty list with a placeholder citation
def _fill_uncited_d2a(val):
    if isinstance(val, list):
        return val if len(val) > 0 else ['D2A-pKa (uncited)']
    s = str(val).strip()
    if s in ('', '[]', 'nan', 'None'):
        return "['D2A-pKa (uncited)']"
    return val
d2a_oxygen_['citation'] = d2a_oxygen_['citation'].apply(_fill_uncited_d2a)

# combine only the external test sets from epik paper and Settimo et al. and amino acids 
combined_prefix = pd.concat([comp_9['prefix'], az['prefix'], manchester['prefix'], vertex['prefix'], morgan['prefix'], comp_ab['prefix'], aa_df['prefix']], axis=0)
combined_smiles = pd.concat([comp_9['smiles'], az['smiles'], manchester['smiles'], vertex['smiles'], morgan['smiles'], comp_ab['smiles'], aa_df['smiles']], axis=0)
combined_micropka = pd.concat([comp_9['micropka input'], az['micropka input'], manchester['micropka input'], vertex['micropka input'], morgan['micropka input'], comp_ab['micropka input'], aa_df['micropka input']], axis=0)
combined_pka = pd.concat([comp_9['target'], az['target'], manchester['target'], vertex['target'], morgan['target'], comp_ab['target'], aa_df['target']], axis=0)
combined_citations = pd.concat([comp_9['citation'], az['citation'], manchester['citation'], vertex['citation'], morgan['citation'], comp_ab['citation'], aa_df['citation']], axis=0)

combined = pd.concat([combined_prefix, combined_smiles, combined_micropka, combined_pka, combined_citations], axis=1)
combined['canonical smiles']  = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in combined['smiles']]

print('# of molecules in combined df with all other datasets other than ML meets pKa: ' + str(len(combined)))

# get fingerprints of ML meets pKa and of the combined epik external test sets 
combined = mol_and_fp(combined, 'canonical smiles')

print('================================================= LOOKING FOR OVERLAP BETWEEN DATASETS ==================================================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
comp_9_az_matches = find_micropka_tanimoto_overlaps(comp_9, az, 1.00)
comp_9_aa_matches = find_micropka_tanimoto_overlaps(comp_9, aa_df, 1.00)
comp_9_manchester_matches = find_micropka_tanimoto_overlaps(comp_9, manchester, 1.00)
comp_9_vertex_matches = find_micropka_tanimoto_overlaps(comp_9, vertex, 1.00)
comp_9_morgan_matches = find_micropka_tanimoto_overlaps(comp_9, morgan, 1.00)
comp_9_ab_matches = find_micropka_tanimoto_overlaps(comp_9, comp_ab, 1.00)
az_manchester_matches = find_micropka_tanimoto_overlaps(az, manchester, 1.00)
az_aa_matches = find_micropka_tanimoto_overlaps(az, aa_df, 1.00)
az_vertex_matches = find_micropka_tanimoto_overlaps(az, vertex, 1.00)
az_morgan_matches = find_micropka_tanimoto_overlaps(az, morgan, 1.00)
az_comp_ab_matches = find_micropka_tanimoto_overlaps(az, comp_ab, 1.00)
manchester_vertex_matches = find_micropka_tanimoto_overlaps(manchester, vertex, 1.00)
manchester_aa_df_matches = find_micropka_tanimoto_overlaps(manchester, aa_df, 1.00)
manchester_morgan_matches = find_micropka_tanimoto_overlaps(manchester, morgan, 1.00)
manchester_comp_ab_matches = find_micropka_tanimoto_overlaps(manchester, comp_ab, 1.00)
vertex_morgan_matches = find_micropka_tanimoto_overlaps(vertex, morgan, 1.00)
vertex_aa_matches = find_micropka_tanimoto_overlaps(vertex, aa_df, 1.00)
vertex_comp_ab_matches = find_micropka_tanimoto_overlaps(vertex, comp_ab, 1.00)
morgan_comp_ab_matches = find_micropka_tanimoto_overlaps(morgan, comp_ab, 1.00)
morgan_aa_matches = find_micropka_tanimoto_overlaps(morgan, aa_df, 1.00)
aa_comp_ab_matches = find_micropka_tanimoto_overlaps(aa_df, comp_ab, 1.00)

# see if there is overlap with molecules in ML meets pKa and external data from epik paper and settimo et al paper 
ml_all_matches = find_micropka_tanimoto_overlaps(ml_meets_pka, combined, 1.00)
print('# of overlapping molecules between ML meets pKa and all other datasets: ' + str(len(ml_all_matches)))

# print overlaps 
overlaps = [('Comparison of 9 programs for pKa prediction','AZ', len(comp_9_az_matches)),
            ('Comparison of 9 programs for pKa prediction','Amino Acids', len(comp_9_aa_matches)),
            ('Comparison of 9 programs for pKa prediction','Manchester', len(comp_9_manchester_matches)),
            ('Comparison of 9 programs for pKa prediction','Vertex', len(comp_9_vertex_matches)),
            ('Comparison of 9 programs for pKa prediction','Morgan', len(comp_9_morgan_matches)),
            ('Comparison of 9 programs for pKa prediction', 'Comparison of Acidic and Basic pKa', len(comp_9_ab_matches)),
            ('AZ','Manchester', len(az_manchester_matches)),
            ('AZ','Amino Acids', len(az_aa_matches)),
            ('AZ','Vertex', len(az_vertex_matches)),
            ('AZ','Morgan', len(az_morgan_matches)),
            ('AZ','Comparison of Acidic and Basic pKa', len(az_comp_ab_matches)),
            ('Manchester','Vertex', len(manchester_vertex_matches)),
            ('Manchester','Amino Acids', len(manchester_aa_df_matches)),
            ('Manchester','Morgan', len(manchester_morgan_matches)),
            ('Manchester','Comparison of Acidic and Basic pKa', len(manchester_comp_ab_matches)),
            ('Vertex','Morgan', len(vertex_morgan_matches)),
            ('Vertex','Amino Acids', len(vertex_aa_matches)),
            ('Vertex','Comparison of Acidic and Basic pKa', len(vertex_comp_ab_matches)),
            ('Morgan','Comparison of Acidic and Basic pKa', len(morgan_comp_ab_matches)),
            ('Morgan','Amino Acids', len(morgan_aa_matches)),
            ('Amino Acids','Comparison of Acidic and Basic pKa', len(aa_comp_ab_matches))]
overlaps_df = pd.DataFrame(overlaps, columns=['Dataset 1', 'Dataset 2','Overlap Count'])
overlaps_matrix = (overlaps_df.pivot(index='Dataset 1', columns='Dataset 2', values='Overlap Count').fillna(0).astype(int))
print(overlaps_matrix)
combined_list = (
    # comp_9 vs everyone
    comp_9_az_matches + comp_9_aa_matches + comp_9_manchester_matches + comp_9_vertex_matches + comp_9_morgan_matches + comp_9_ab_matches +
    # ml_meets_pKa “matches”
    ml_all_matches +
    # az vs others
    az_aa_matches + az_manchester_matches + az_vertex_matches + az_morgan_matches + az_comp_ab_matches +
    # manchester vs others
    manchester_vertex_matches + manchester_aa_df_matches + manchester_morgan_matches  + manchester_comp_ab_matches +
    # vertex vs others
    vertex_morgan_matches + vertex_aa_matches + vertex_comp_ab_matches +
    # morgan vs others
    morgan_comp_ab_matches + morgan_aa_matches +
    # aa vs comp_ab
    aa_comp_ab_matches
)
print(str(len(combined_list)) + ' molecules overlap between these ^ datasets')
print('============================== AVERAGING STEP: average pKas from different datasets if within 0.5 difference ==============================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')

# put all the molecules that have overlap with each other into one df 
average_these = pd.DataFrame(combined_list)
all_match = average_these['FP_prot_1'].eq(average_these['FP_prot_2']).all()
print('Sanity check: Do the fingerprints match of all ' + str(len(average_these)) + ' overlapping molecules? ' + str(all_match))
print(str(len(average_these)) + ' molecules need to be averaged from the above step ^')
all_match = average_these['FP_prot_1'].eq(average_these['FP_prot_2']).all()
print('Sanity check: Do the fingerprints match of all ' + str(len(average_these)) + ' overlapping molecules? ' + str(all_match))

# use IUPAC to see any overlap with this^ df 
matches = []
average_these = average_these.rename(columns={'FP_prot_1':'FP'})
average_these[['SMILES_1','deprot_1']] = average_these['micropka_input_1'].str.split('>>',expand=True)
average_these[['SMILES_2','deprot_2']] = average_these['micropka_input_2'].str.split('>>',expand=True)
for idx1, row1 in average_these.iterrows():
    prefix2 = list(iupac['prefix'])
    sims = DataStructs.BulkTanimotoSimilarity(row1['FP'], iupac['FP'].to_list())
    for idx2, sim in enumerate(sims):
        if sim >= 1.00 and row1['prefix'] == prefix2[idx2]:
            matches.append({
                'prefix'          : row1['prefix'], 
                'SMILES_1'        : row1['SMILES_1'], 
                'micropka_input_1': row1['micropka_input_1'],
                'pKa_1'           : row1['pKa_1'], 
                'citation_1'      : row1['citation_1'],
                'FP_1'            : row1['FP'],
                'SMILES_2'        : row1['SMILES_2'], 
                'micropka_input_2': row1['micropka_input_2'],
                'pKa_2'           : row1['pKa_2'],  
                'citation_2'      : row1['citation_2'],
                'iupac smiles'    : iupac['canonical smiles'].iloc[idx2], 
                'pKa_3'           : iupac['pka_value'].iloc[idx2],
                'citation_3'      : iupac['citation'].iloc[idx2],
                'FP_3'            : iupac['FP'].iloc[idx2],
                'Tanimoto'        : sim
                })
ml_external_epik_iupac_overlap = pd.DataFrame(matches) 
print('Out of the ' + str(len(average_these)) + ' molecules needing to be averaged ' + str(len(ml_external_epik_iupac_overlap)) + ' molecules overlap with IUPAC')
all_match_ = ml_external_epik_iupac_overlap['FP_1'].eq(ml_external_epik_iupac_overlap['FP_3']).all()
print('Sanity check: Do the fingerprints match of all ' + str(len(ml_external_epik_iupac_overlap)) + ' overlapping molecules? ' + str(all_match_))

# drop any of the overlapping molecules from ml_external_epik_iupac_overlap that are in average_these df 
average_these = average_these[~average_these['FP'].isin(ml_external_epik_iupac_overlap['FP_1'].unique())].copy()
average_these = average_these.rename(columns={'FP':'FP_1','FP_prot_2':'FP_2'})
# combine all the two dfs with averaged pkas
together_average_pkas = pd.concat([average_these, ml_external_epik_iupac_overlap], axis=0).reset_index(drop=True)
print('# of molecules with averaged pKa BEFORE cleaning out unreliable pKas: ' + str(len(together_average_pkas)))
# get averages, mean absolute deviation, and organize df with only information needed 
avg_, mad_, cit_, pkas_, diff_ = [], [], [], [], []
for i, row in together_average_pkas.iterrows():
    if pd.notna(row['pKa_3']):
        pkas = [row['pKa_1'], row['pKa_2'], row['pKa_3']]
        cit = [row['citation_1'], row['citation_2'], row['citation_3']]
        avg = (row['pKa_1'] + row['pKa_2'] + row['pKa_3']) / len(pkas)
        avg_.append(round(avg,2))
        pkas_.append(pkas)
        cit_.append(cit)
         # mean absolute deviation # cutoff would be 0.25 bc MAD = 1/2 |a-b|
        mad = sum(abs(x - avg) for x in pkas) / len(pkas)
        mad_.append(round(mad,2))
        diff_.append(np.nan)
    else:
        pkas = [row['pKa_1'], row['pKa_2']]
        cit = [row['citation_1'], row['citation_2']]
        avg = (row['pKa_1'] + row['pKa_2']) / len(pkas)
        avg_.append(round(avg,2))
        pkas_.append(pkas)
        cit_.append(cit)
        # mean absolute deviation # cutoff would be 0.25 bc MAD = 1/2 |a-b|
        mad = sum(abs(x - avg) for x in pkas) / len(pkas)
        mad_.append(round(mad,2))
        # calc absolute difference 
        diff = abs(pkas[0] - pkas[1])
        diff_.append(round(diff,2))
together_average_pkas = together_average_pkas.drop(['pKa_1','pKa_2','pKa_3','citation_1','citation_2','citation_3','FP_2','FP_3','iupac smiles',
'idx1','idx2','half','FP_1','Tanimoto_prot','Tanimoto_deprot','FP_deprot_1','FP_deprot_2'], axis=1)
together_average_pkas = together_average_pkas.assign(**{'pKas': pkas_,'average':avg_,'mean absolute error': mad_,'absolute difference': diff_,'citations': cit_})

# remove molecules with an absolute difference of more than 0.5 and / or with a mean absolute deviation of more than 0.25
mask_add = ((together_average_pkas['absolute difference'] > 0.5) |(together_average_pkas['mean absolute error'] > 0.25))
unreliable_pka = together_average_pkas[mask_add].copy()
# mask for rows where abs diff and MAE == 0.0 because these shouldn't be accounted as averages anyways
mask_no_diff = ((together_average_pkas['absolute difference'] == 0.0) &(together_average_pkas['mean absolute error'] == 0.0))
# update pKas → keep only first element, still inside list
together_average_pkas.loc[mask_no_diff, "pKas"] = (together_average_pkas.loc[mask_no_diff, "pKas"].apply(lambda x: [x[0]] if isinstance(x, (list, tuple)) and len(x) > 0 else [np.nan]))
together_average_pkas.loc[mask_no_diff, "Tanimoto"] = np.nan
# remove molecules with the same pKas from various source and make new df with them; will need to move them to other dataset
no_diff_mae = together_average_pkas[mask_no_diff].copy()
no_diff_mae["pKas"] = no_diff_mae["pKas"].apply(lambda x: [x[0]] if isinstance(x, (list, tuple)) and len(x) > 0 else [np.nan])
together_average_pkas = together_average_pkas[~mask_no_diff].copy()
# remove unreliable pKas and their molecules from dataset 
len_together_average_pkas = str(len(together_average_pkas))
_together_average_pkas = together_average_pkas.copy() # copy to be used later when placing back the molecules used in Reaxys database
together_average_pkas = together_average_pkas.drop(index=unreliable_pka.index).reset_index(drop=True)
print('# of molecules with unreliable pKas: ' + str(len(unreliable_pka)))
print('SMILES of unreliable pKas: ' + str(unreliable_pka['SMILES_1'].values))
print('# of molecules with averaged pKa BEFORE|AFTER cleaning out unreliable pKas: ' + str(len_together_average_pkas) + '|' + str(len(together_average_pkas)))

# Reaxys information for unreliable molecules; all molecules from reaxys needed to be in H2O solution and 25 degrees C
reaxys0 = [{'i':24,'prefix':'basic','SMILES':'CC(CN(C)C)CN1c2ccccc2Sc2ccccc21','pKas':[9.235, 8.5],'average':8.87,'citations':['Machine Learning meets pKa', 'Vertex'],
'reaxys_dissociation_exponent_1':'a1/apparent','reaxys_pKa_1':8.6,'reaxys_dissociation_group_1':'','reaxys_citation_1':['Ogiso,Iwaki,Kuranari; Chemical and Pharmaceutical Bulletin, 1983 [Reaxys]'],'reaxys_solvent_1':'','Drop':'N'}] # Vertex citation agrees with Reaxys 

reaxys1 = [{'i':25,'prefix':'basic','SMILES':'CC(CN(C)C)CN1c2ccccc2Sc2ccccc21','pKas':[9.235, 8.5],'average':8.87,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'a1/apparent','reaxys_pKa_1':8.6,'reaxys_dissociation_group_1':'','reaxys_citation_1':['Ogiso,Iwaki,Kuranari; Chemical and Pharmaceutical Bulletin, 1983[Reaxys]'],'reaxys_solvent_1':'','Drop':'Y'}] # same as reaxys1 ^^^^; keep 8.5 pKa

reaxys2 = [{'i':31,'prefix':'basic','SMILES':'CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1','pKas':[9.203333, 10.0],'average':9.6,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':8.45,'reaxys_citation_1':['Bergstrom, Christel A.S.,Strafford,Lazorova,Avdeef,Luthman,Artursson; Journal of Medicinal Chemistry, 2003[Reaxys]'],'reaxys_solvent_1':'H2O','Drop':'N'}] # combine all values and see if it changes the MAE

reaxys3 = [{'i':40,'prefix':'basic','SMILES':'CNC(=O)Oc1ccc2c(c1)[C@@]1(C)CCN(C)[C@H]1N2C','pKas':[6.12, 8.17],'average':7.14,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys4 = [{'i':45,'prefix':'basic','SMILES':'COC(=O)[C@@H]1[C@H](O)C[C@@H]2CC[C@H]1N2C','pKas':[8.85, 8.15],'average':8.5,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys5 = [{'i':52,'prefix':'basic','SMILES':'COc1cc2c(cc1OC)[C@]13CCN4CC5=CCO[C@@H]6CC(=O)N2[C@@H]1[C@@H]6[C@@H]5C[C@@H]43','pKas':[5.86, 8.28],'average':7.07,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys6 = [{'i':67,'prefix':'basic','SMILES':'CN(C)CCC=C1c2ccccc2C=Cc2ccccc21','pKas':[9.69, 8.241],'average':8.97,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys7 = [{'i':68,'prefix':'basic', 'SMILES':'CN(C)CCC=C1c2ccccc2CCc2ccccc21','pKas':[9.42, 8.2],'average':8.81,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'], # keep ML meets pKa 
'reaxys_dissociation_exponent_1':'a1/thermodynamic','reaxys_pKa_1':9.4,'reaxys_citation_1':['Sugawara,Takekuma,Yamada,Kobayashi,Iseki,Katsumi;Journal of Pharmaceutucal Sciences, 1998[Reaxys]'],'reaxys_solvent_1':'','Drop':'N'}]

reaxys8 = [{'i':75,'prefix':'basic','SMILES':'CN(C/C=C/C#CC(C)(C)C)Cc1cccc2ccccc12','pKas':[8.5, 7.12],'average':7.81,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys9 = [{'i':94,'prefix':'basic','SMILES':'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21','pKas':[9.23, 8.0],'average':8.62,'citations':['Machine Learning meets pKa', 'Vertex'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}]# not seen in Reaxys; drop

reaxys10 = [{'i':95,'prefix':'basic','SMILES':'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21','pKas':[9.23, 8.0],'average':8.62,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys11 = [{'i':109,'prefix':'basic','SMILES':'O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1','pKas':[7.965, 7.45],'average':7.71,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys12 = [{'i':119,'prefix':'acidic','SMILES':'CC(=O)C[C@@H](c1ccccc1)c1c(O)c2ccccc2oc1=O','pKas':[6.655, 5.02],'average':5.84,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # no results on Reaxys; drop

reaxys13 = [{'i':122,'prefix':'acidic','SMILES':'CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1','pKas':[4.353333, 3.8],'average':4.08,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'], # Compar of Acidic seems to be for S-Ketoprofen
'reaxys_dissociation_exponent_1':'a1/apparent','reaxys_pKa_1':4.25,'reaxys_citation_1':['Bouchard, Carrupt, Testa, Gobry, Girault; Chemistry-A European Journal, 2002[Reaxys]'],'reaxys_solvent_1':'H2O','reaxys_method_1':'potentiometric',
'reaxys_dissociation_exponent_2':'a1/apparent','reaxys_pKa_2':3.89,'reaxys_citation_2':['Winiwarter, Bonham, Ax, Hallberg, Lennernas, Karlen; Journal of Medicinal Chemistry, 1998[Reaxys]'],'reaxys_solvent_2':'H2O','reaxys_method_2':'potentiometric','Drop':'N*'}]

reaxys14 = [{'i':123,'prefix':'acidic','SMILES':'CC(C)Cc1ccc(C(C)C(=O)O)cc1','pKas':[5.205714, 4.44],'average':4.82,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'], # combine all pKas and see if the MAE changes 
'reaxys_dissociation_exponent_1':'b1/apparent','reaxys_pKa_1':4.6,'reaxys_citation_1':['Wiedenbeck,Gebauer,Colfen; Analytical Chemistry, 2020[Reaxys]'],'reaxys_solvent_1':'H2O','reaxys_method_1':'potentiometric',
'reaxys_dissociation_exponent_2':'a1/apparent','reaxys_pka_2':4.31,'reaxys_citation_2':['Bouchard, Carrupt, Testa,Gobry,Girault; Chemistry-A European Journal, 2002[Reaxys]'],'reaxys_solvent_2':'H2O','reaxys_method_2':'potentiometric',
'reaxys_dissociation_exponent_3':'a1/apparent','reaxys_pka_3':4.4,'reaxys_citation_3':['Oumada,Raaandfols,Roseands,Bosch;Journal of Pharmaceutrical Sciences, 2002[Reaxys]'],'reaxys_solvent_3':'H2O','reaxys_method_3':'potentiometric','Drop':'N'}]

reaxys15 = [{'i':124,'prefix':'acidic','SMILES':'CC(C)Cc1ccc([C@H](C)C(=O)O)cc1','pKas':[5.2, 4.44],'average':4.82,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # repeat from above; drop

reaxys16 = [{'i':125,'prefix':'acidic','SMILES':'CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1','pKas':[4.51, 5.15],'average':4.83,'citations':['Machine Learning meets pKa', 'Vertex'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys17 = [{'i':126,'prefix':'acidic','SMILES':'CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1','pKas':[4.51, 5.15],'average':4.82,'citations':['Machine Learning meets pKa', 'Comparison of Acidic and Basic pKa Settimo et al.'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # repeat from above; drop

reaxys18 = [{'i':137,'prefix':'basic','SMILES':'Cc1c(N(C)C)c(=O)n(-c2ccccc2)n1C','pKas':[4.48, 5.06],'average':4.77,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'','reaxys_pKa_1':'','reaxys_citation_1':'','reaxys_solvent_1':'','Drop':'Y'}] # not seen in Reaxys; drop

reaxys19 = [{'i':150,'prefix':'acidic','SMILES':'O=C(O)/C=C\\c1ccccc1','pKas':[3.88, 4.45],'average':4.17,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'a/apparent','reaxys_pKa_1':4.27,'reaxys_citation_1':['Hoek;Svensk ZKemisk Tidskridft,1953[Reaxys]'],'reaxys_solvent_1':'H2O','reaxys_method_1':'potentiometric',
'reaxys_dissociation_exponent_2':'thermodynamic','reaxys_pKa_2':4.44,'reaxys_citation_2':['Dippy;Lewis;Journal of the Chemical Society, 1937[Reaxys]'],'reaxys_solvent_2':'H2O','reaxys_method_2':'',
'reaxys_dissociation_exponent_3':'thermodynamic','reaxys_pKa_3':4.42,'reaxys_citation_3':['Roth;Stoermer;Chemische Beriche, 1913[Reaxys]'],'reaxys_solvent_3':'H2O','reaxys_method_3':'',
'reaxys_dissociation_exponent_4':'thermodynamic','reaxys_pKa_3':4.43,'reaxys_citation_4':['White;Jones;American Chemical Journal, 1910[Reaxys]'],'reaxys_solvent_4':'H2O','reaxys_method_4':'','Drop':'N'}]

reaxys20 = [{'i':156,'prefix':'acidic','SMILES':'O=C(O)c1ccccc1O','pKas':[3.354167, 2.78],'average':3.07,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'a1/apparent','reaxys_pKa_1':2.853,'reaxys_citation_1':['Farajtabar,Gharib;Monashefte fur Chemie, 2009[Reaxys]'],'reaxys_solvent_1':'H2O','reaxys_method_1':'',
'reaxys_dissociation_exponent_2':'a1/apparent','reaxys_pKa_2':2.77,'reaxys_citation_2':['Aksoy,Oezer; Chemical and Pharmaceutical Bulletin, 2004[Reaxys]'],'reaxys_solvent_2':'H2O','reaxys_method_2':'potentiometric',
'reaxys_dissociation_exponent_3':'a1/apparent','reaxys_pKa_3':3.156,'reaxys_citation_3':['Degim,Akay; Il Farmaco, 2001[Reaxys]'],'reaxys_solvent_3':'H2O','reaxys_method_3':'',
'reaxys_dissociation_exponent_4':'a1/apparent','reaxys_pKa_4':3,'reaxys_citation_4':['Mchedlov-Petrossyan,Mayorga; Journal of the Chemical Society-Faraday Transactions, 1992[Reaxys]'],'reaxys_solvent_4':'H2O','reaxys_method_4':'spectrophotometric',
'reaxys_dissociation_exponent_5':'a1/apparent','reaxys_pKa_5':2.99,'reaxys_citation_5':['Papadopoulos,Avranas; Journal of Solution Chemistry, 1991[Reaxys]'],'reaxys_solvent_5':'H2O','reaxys_method_5':'conductometric',
'reaxys_dissociation_exponent_6':'a1/apparent','reaxys_pKa_6':3.1,'reaxys_citation_6':['Dhat,Jahagirdar;Indian Journal of Chemistry, Section A:Inorganic, Physcial, Theoretical and Analytical, 1982[Reaxys]'],'reaxys_solvent_6':'H2O','reaxys_method_6':'potentiometric',
'reaxys_dissociation_exponent_7':'a1/apparent','reaxys_pKa_7':2.97,'reaxys_citation_7':['Chattopadhyaya,Singh;Indian Journal of Chemistry, Section A: Inorganic, Physical, Theoretical and Analytical, 1980[Reaxys]'],'reaxys_solvent_7':'H2O','reaxys_method_7':'spectrophotometric',
'reaxys_dissociation_exponent_8':'a1/apparent','reaxys_pKa_8':3,'reaxys_citation_8':['Bray et al.;Journal of the Chemical Society, 1957[Reaxys]'],'reaxys_solvent_8':'H2O','reaxys_method_8':'conductometric',
'reaxys_dissociation_exponent_9':'1/themodynamic','reaxys_pKa_9':2.98,'reaxys_citation_9':['Minnick;Journal of Physical Chemistry, 1939[Reaxys]'],'reaxys_solvent_9':'H2O','reaxys_method_9':'',
'reaxys_dissociation_exponent_10':'thermodynamic','reaxys_pKa_10':2.97,'reaxys_citation_10':['White;American Chemical Journal, 1910[Reaxys]'],'reaxys_solvent_10':'H2O','reaxys_method_10':'','Drop':'N'}]

reaxys21 = [{'i':157,'prefix':'acidic','SMILES':'O=C(O)c1ccccn1','pKas':[5.32, 0.99],'average':3.16,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction'],
'reaxys_dissociation_exponent_1':'a1/apparent','reaxys_pKa_1':1.69,'reaxys_citation_1':['Landaeta,Barrera,del Carpio,Nobrega,Rodiguez,Coll,David S.,Lubes;Inorganica Chimica Acta,2018[Reaxys]'], 'reaxys_solvent_1':'H2O','reaxys_method_1':'potentiometric',
'reaxys_dissociation_exponent_2':'a2/apparent','reaxys_pKa_1':5.34,'reaxys_citation_2':['Landaeta,Barrera,del Carpio,Nobrega,Rodiguez,Coll,David S.,Lubes;Inorganica Chimica Acta,2018[Reaxys]'], 'reaxys_solvent_2':'H2O','reaxys_method_2':'potentiometric',
'reaxys_dissocitation_exponent_3':'b1/apparent','reaxys_pKa_2':5.07,'reaxys_citation_3':['Centeno,Martinez,Araujo,Brit,Del Carpio,Hernadez,Lubes;Journal of Solution Chemistry, 2014[Reaxys]'],'reaxys_solvent_3':'H2O','reaxys_method_3':'potentiometric','Drop':'N*'}] # 5 pKa is pKaH and 0.99 pka 

reaxys22 = [{'i':164,'prefix':'acidic','SMILES':'CCCCNc1cc(C(=O)O)cc(S(N)(=O)=O)c1Oc1ccccc1','pKas':[6.07, 3.6],'average':4.83,'citations':['AZ', 'Comparison of Acidic and Basic pKa Settimo et al.'],'Drop':'Y'}] # no results with only H2O solvent in Reaxys; drop

reaxys23 = [{'i':170,'prefix':'basic','SMILES':'CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCN[C@H](C)C3)c(F)c21','pKas':[8.87, 5.742],'average':7.31,'citations':['AZ', 'Comparison of Acidic and Basic pKa Settimo et al.'],'Drop':'Y'}] # no results on Reaxys; drop

reaxys24 = [{'i':199,'prefix':'basic','SMILES':'Cc1cccc(-n2ncc(C(=O)Nc3nccs3)c2C2CCNCC2)c1','pKas':[9.5, 10.5],'average':10.0,'citations':['Vertex', 'Comparison of Acidic and Basic pKa Settimo et al.'],'Drop':'Y'}] # no results on Reaxys; drop

reaxys25 = [{'i':200,'prefix':'acidic','SMILES':'Cc1cccc(-n2ncc(C(=O)Nc3nccs3)c2C2CCNCC2)c1','pKas':[10.5, 9.5],'average':10.0,'citations':['Vertex', 'Comparison of Acidic and Basic pKa Settimo et al.'],'Drop':'Y'}] # no results on Reaxys; drop

reaxys26 = [{'i':224,'prefix':'acidic','SMILES':'O=C(O)c1cccnc1','pKas':[2.05, 2.4, 4.835],'average':3.1,'citations':['Comparison of 9 programs for pKa prediction', 'Comparison of Acidic and Basic pKa Settimo et al.', 'IUPAC'],
'reaxys_dissociation_exponent_1':'a1/apparent','reaxys_dissociation_group':'COO(1-)','reaxys_pKa_1':1.96,'reaxys_citation_1':['Garcia,Ibeas,Leal;Journal of Physical Organic Chemistry, 1996[Reaxys]'],'reaxys_solvent_1':'H2O','reaxys_method_1':'potentiometric',
'reaxys_dissociation_exponent_2':'a2/apparent','reaxys_dissociation_group':'NH(1+)','reaxys_pKa_2':5,'reaxys_citation_2':['Garcia,Ibeas,Leal;Journal of Physical Organic Chemistry, 1996[Reaxys]'],'reaxys_solvent_2':'H2O','reaxys_method_2':'spectrophotometric','Drop':'N*'}] # 5 is pKaH and 2 is pKa

reaxys27 = [{'i':226,'prefix':'acidic','SMILES':'Cn1c(=O)c2[nH]cnc2n(C)c1=O','pKas':[7.71, 8.55, 8.59],'average':8.28,'citations':['Machine Learning meets pKa', 'Comparison of 9 programs for pKa prediction', 'IUPAC'],'Drop':'Y'}] # no results on Reaxys; drop
# combine reaxys1 through reaxys27
reaxys = [globals().get(f'reaxys{i}') for i in range(0,28)] 
reaxys_ = [entry for sublist in reaxys if sublist for entry in sublist]
reaxys_df = pd.DataFrame(reaxys_)
# drop any of the molecules we confirmed with Reaxys to drop: have 'Y' in 'Drop' column
reaxys_df = reaxys_df[ reaxys_df['Drop'] != 'Y' ].reset_index(drop=True)
# save any of the rows with 'N*' as a separate df as we'll need to deal with them later
nstar_df = reaxys_df[ reaxys_df['Drop'] == 'N*' ].copy()
# focus only on the molecules that we will not drop and just need to average with the new values from Reaxys 
reaxys_df = reaxys_df[reaxys_df['Drop'] == 'N'].reset_index(drop=True)
pka_cols = [f'reaxys_pKa_{i}' for i in range(1, 11)] # get the pKas associated with Reaxys database 
citation_cols = [f'reaxys_citation_{i}'  for i in range(1, 11)] # get the citations given from Reaxys from ^these pKas 
new_pkas, new_avgs, new_mads, new_citations = [], [], [], []
for _, row in reaxys_df.iterrows(): # get the new averages based on the newly gotten values from Reaxys and also update citations and calc mean absolute error 
    vals = []
    orig = row['pKas']
    if isinstance(orig, (list, tuple)):
        vals += [float(x) for x in orig if pd.notnull(x)]
    # add any reaxys_pKa_N
    for col in pka_cols:
        v = row.get(col)
        if pd.notnull(v):
            vals.append(float(v))
    # calc mean absolute error
    mean_val = np.mean(vals)
    mad_val = np.mean([abs(x - mean_val) for x in vals])
    new_pkas.append(vals)
    new_avgs.append(mean_val)
    new_mads.append(mad_val)
    # --- combine citations ---
    cits = []
    orig_c = row['citations']
    # parse original citations (if string)
    if isinstance(orig_c, str):
        try:
            orig_list = ast.literal_eval(orig_c)
        except Exception:
            orig_list = [orig_c]
    else:
        orig_list = list(orig_c) if orig_c is not None else []
    cits += orig_list
    # add any reaxys_citation_N
    for col in citation_cols:
        v = row.get(col)
        if pd.notnull(v):
            if isinstance(v, str) and v.startswith('['):
                try:
                    parsed = ast.literal_eval(v)
                    cits.extend(parsed)
                except Exception:
                    cits.append(v)
            elif isinstance(v, (list, tuple)):
                cits.extend(v)
            else:
                cits.append(v)
    # dedupe while preserving order
    seen = set()
    dedup = []
    for entry in cits:
        if entry not in seen:
            seen.add(entry)
            dedup.append(entry)
    new_citations.append(dedup)
# update columns 
reaxys_df = reaxys_df.assign(pKas=new_pkas, average=new_avgs, **{'mean absolute error': new_mads},citations=new_citations)
reaxys_df = reaxys_df.loc[
    reaxys_df['mean absolute error'] <= 0.25
].reset_index(drop=True) # keep only the molecules with mean absolute error of less than 0.25
breakpoint()
# update the copy of together_average_pkas with the new citations and averages based on what was gathered in Reaxys
cols = ['pKas','citations','mean absolute error']
_together_average_pkas.loc[reaxys_df['i'], cols] = (reaxys_df.set_index('i')[cols].loc[_together_average_pkas.index.intersection(reaxys_df['i']), cols])
subset = _together_average_pkas.loc[reaxys_df['i'].tolist()] # get only the rows of interest 
together_average_pkas = pd.concat([together_average_pkas, subset], axis=0) # add these rows into together_average_pkas 
# revisit nstar_df 
micropka_input_122 =  {'prefix':'acidic','SMILES_1':nstar_df['SMILES'].iloc[0],'micropka_input_1':'CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1>>CC(C(=O)[O-])c1cccc(C(=O)c2ccccc2)c1','Tanimoto':1,'pKas':[nstar_df['pKas'].iloc[0][0],nstar_df['reaxys_pKa_1'].iloc[0]],'average':np.mean([nstar_df['pKas'].iloc[0][0],nstar_df['reaxys_pKa_1'].iloc[0]]),
'mean absolute error':'','absolute difference':'','citations':[nstar_df['citations'].iloc[0][0],nstar_df['reaxys_citation_1'].iloc[0][0]]}# account for S-Ketoprofen 
S_ketoprofen = Chem.MolToSmiles(inchi.MolFromInchi('InChI=1S/C16H14O3/c1-11(16(18)19)13-8-5-9-14(10-13)15(17)12-6-3-2-4-7-12/h2-11H,1H3,(H,18,19)/t11-/m0/s1'), canonical=True)
micropka_input_122_S = {'prefix':'acidic','SMILES_1':nstar_df['SMILES'].iloc[0],'micropka_input_1':str(S_ketoprofen)+'>>C[C@H](C(=O)[O-])c1cccc(C(=O)c2ccccc2)c1','Tanimoto':1,'pKas':[nstar_df['pKas'].iloc[0][1],nstar_df['reaxys_pKa_2'].iloc[0]],'average':np.mean([nstar_df['pKas'].iloc[0][1],nstar_df['reaxys_pKa_2'].iloc[0]]),
'mean absolute error':'','absolute difference':'','citations':[nstar_df['citations'].iloc[0][1],nstar_df['reaxys_citation_2'].iloc[0][0]]} # account for S-Ketoprofen
micropka_input_157_pkaH = {'prefix':'basic','SMILES_1':nstar_df['SMILES'].iloc[1],'micropka_input_1':'O=C(O)c1cccc[nH+]1>>'+str(nstar_df['SMILES'].iloc[1]),'Tanimoto':1,'pKas':[nstar_df['pKas'].iloc[1][0],nstar_df['reaxys_pKa_1'].iloc[1],nstar_df['reaxys_pKa_2'].iloc[1]],'average':np.mean([nstar_df['pKas'].iloc[1][0],nstar_df['reaxys_pKa_1'].iloc[1],nstar_df['reaxys_pKa_2'].iloc[1]]),
'mean absolute error':'','absolute difference':'','citations':[nstar_df['citations'].iloc[1][0],nstar_df['reaxys_citation_1'].iloc[1][0],nstar_df['reaxys_citation_2'].iloc[1][0]]} # moved 157 pKa value into reaxys_mols below 
micropka_input_224 = {'prefix':'acidic','SMILES_1':nstar_df['SMILES'].iloc[2],'micropka_input_1':'O=C(O)c1ccccn1>>O=C([O-])c1ccccn1','Tanimoto':1,'pKas':[nstar_df['pKas'].iloc[2][0], nstar_df['pKas'].iloc[2][1],nstar_df['reaxys_pKa_1'].iloc[2]],'average':np.mean([nstar_df['pKas'].iloc[2][0], nstar_df['pKas'].iloc[2][1],nstar_df['reaxys_pKa_1'].iloc[2]]),
'mean absolute error':'','absolute difference':'','citations':[nstar_df['citations'].iloc[2][0], nstar_df['citations'].iloc[2][1],nstar_df['reaxys_citation_1'].iloc[2][0]]}
micropka_input_224_pkaH = {'prefix':'basic','SMILES_1':nstar_df['SMILES'].iloc[2],'micropka_input_1':'O=C(O)c1ccc[nH+]c1>>'+str(nstar_df['SMILES'].iloc[2]),'Tanimoto':1,'pKas':[nstar_df['pKas'].iloc[2][2],nstar_df['reaxys_pKa_2'].iloc[2]],'average':np.mean([nstar_df['pKas'].iloc[2][2],nstar_df['reaxys_pKa_2'].iloc[2]]),
'mean absolute error':'','absolute difference':'','citations':[nstar_df['citations'].iloc[2][2],nstar_df['reaxys_citation_2'].iloc[2][0]]}
add_these = [micropka_input_122 , micropka_input_122_S , micropka_input_157_pkaH , micropka_input_224 , micropka_input_224_pkaH]
reaxys_add = pd.DataFrame(add_these)
# adding this 
# same molecule different ionization: changing for sequential ionization; change not necessary due to basic value being an average and multiple sources 
# CCOC(=O)C(CCc1ccccc1)[N+]C(C)C(=O)N1CCCC1C(=O)O>>CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O | pKa=5.31 --> CCOC(=O)C(CCc1ccccc1)[N+]C(C)C(=O)N1CCCC1C(=O)[O-]>>CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)[O-]
# CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O>>CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)[O-] | pKa=2.85 --> CCOC(=O)C(CCc1ccccc1)[N+]C(C)C(=O)N1CCCC1C(=O)O>>CCOC(=O)C(CCc1ccccc1)[N+]C(C)C(=O)N1CCCC1C(=O)[O-]
reaxys_mols = pd.DataFrame([{'prefix':'basic','SMILES_1':'CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O','micropka_input_1':'CCOC(=O)C(CCc1ccccc1)[N+]C(C)C(=O)N1CCCC1C(=O)O>>CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O','Tanimoto':1.0,'pKas':[5.360000133514404,5.26],'average':5.31,'mean absolute error':0.05,'absolute difference':0.10,'citations':['Comparison of 9 programs for pKa prediction','Reaxys']},
{'prefix':'acidic','SMILES_1':'CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O','micropka_input_1':'CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O>>CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)[O-]','Tanimoto':np.nan,'pKas':[2.85],'average':2.85,'mean absolute error':'','absolute difference':'','citations':['Reaxys']},
{'prefix':'acidic','SMILES_1':nstar_df['SMILES'].iloc[1],'micropka_input_1':_together_average_pkas['micropka_input_1'].iloc[157],'Tanimoto':np.nan,'pKas':[nstar_df['pKas'].iloc[1][1]],'average':np.mean([nstar_df['pKas'].iloc[1][1]]),'mean absolute error':'','absolute difference':'','citations':nstar_df['citations'].iloc[1][1]}])
# put back checked (and verified with reaxys) unreliable pKas if any avaliable
together_average_pkas = pd.concat([together_average_pkas, reaxys_mols, reaxys_add], axis=0).reset_index(drop=True)
together_average_pkas = mol_and_fp(together_average_pkas, smiles_column='SMILES_1')
# look for duplicates in together_average_pkas and average again
duplicates_df = together_average_pkas[together_average_pkas.duplicated(subset=["micropka_input_1",'prefix'], keep=False)].copy().sort_values(['micropka_input_1'])
merged = (duplicates_df.groupby("micropka_input_1", as_index=False).agg({"pKas": lambda x: sum(x, []), "citations": lambda x: sum(x, [])})) # make df that combines the pKas and citations
# drop any duplication of citation and its respective pKa to calculate MAE nicely later
clean_pkas = []
clean_cits = []
for i,(pkas,cits) in enumerate(zip(merged['pKas'],merged['citations'])):
    seen = set()
    pkas_out, cits_out = [], []

    for p, c in zip(pkas, cits):
        if c not in seen:       # keep only the first time we see a citation
            pkas_out.append(p)
            cits_out.append(c)
            seen.add(c)

    clean_pkas.append(pkas_out)
    clean_cits.append(cits_out)
merged["pKas"] = clean_pkas
merged["citations"] = clean_cits
# calculate MAE
maes = []
for pkas in merged["pKas"]:
    if not pkas or len(pkas) == 1:
        maes.append(0.0)  # no error if only one value
    else:
        ref = pkas[0]  # first pKa = reference
        diffs = [abs(p - ref) for p in pkas[1:]]
        maes.append(float(np.mean(diffs)))
merged["mean absolute error"] = maes
merged['absolute difference'] = np.nan
# 1) Replace the overlapping columns in together_average_pkas with merged
cols_to_replace = ["pKas", "citations", "mean absolute error", "absolute difference"]
together_average_pkas.update(merged.set_index("micropka_input_1")[cols_to_replace])
# 2) Drop duplicate molecules so each micropka_input_1 only appears once
together_average_pkas = (together_average_pkas.drop_duplicates(subset=["micropka_input_1","prefix"]).reset_index(drop=True))
row_120 = together_average_pkas.loc[[120]] # found only in Reaxys so it can go into main training data
together_average_pkas = together_average_pkas.drop(index=120).reset_index(drop=True)
# add unreliable_pka into together_average_pkas
together_average_pkas = pd.concat([together_average_pkas,unreliable_pka], axis=0)
# drop any repeated unreliable_pka since the original one seen in together_average_pkas has extra reaxys citations
together_average_pkas = together_average_pkas.drop_duplicates(subset=['micropka_input_1'], keep='first') # make sure to keep the version from together_average_pkas
# add index 120 into no_diff_mae to be part of main training and validation 
no_diff_mae = pd.concat([no_diff_mae, row_120], axis=0)
no_diff_mae['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in no_diff_mae['SMILES_1']]
no_diff_mae = mol_and_fp(no_diff_mae, 'canonical smiles')
no_diff_mae = no_diff_mae.rename(columns={'citations':'citation','average':'target','micropka_input_1':'micropka input'})
no_diff_mae = no_diff_mae.drop(['micropka_input_2','Tanimoto','SMILES_1','SMILES_2','deprot_1','deprot_2','pKas','mean absolute error','absolute difference'],axis=1)

# ----- collapse n-way overlaps in no_diff_mae -----
# PHILIPP ADDED THIS!
# When 3+ sources report identical pKa for the same molecule, the pairwise Tanimoto
# matcher produces (N choose 2) averaged rows. Group by (prefix, micropka input)
# and merge their citation lists into a single row.
_other_cols = [c for c in no_diff_mae.columns if c not in ('prefix', 'micropka input', 'citation')]
_agg_map = {c: 'first' for c in _other_cols}
_agg_map['citation'] = lambda x: list(dict.fromkeys(sum(x, [])))  # concat + dedupe, preserves order

_before_n_diff = len(no_diff_mae)
no_diff_mae = (
    no_diff_mae
    .groupby(['prefix', 'micropka input'], as_index=False, sort=False)
    .agg(_agg_map)
)
print(f"no_diff_mae: collapsed {_before_n_diff - len(no_diff_mae)} rows from n-way overlap pairwise averaging")

breakpoint()
print('# of AVERAGED molecules + added ' + str(np.sum(len(add_these) + len(reaxys_mols) + len(subset))) + ' molecules while cross-checking with Reaxys: ' + str(len(together_average_pkas)))
# combine the other datasets with ML meets pKa 
combined_trimmed = combined[['prefix', 'target', 'canonical smiles', 'citation', 'FP','micropka input']]
ml_meets_pka_trimmed = ml_meets_pka[['prefix', 'target', 'canonical smiles', 'citation', 'FP','micropka input']]
new_data = pd.concat([combined_trimmed, ml_meets_pka_trimmed], ignore_index=True)
#new_data = pd.concat([new_data,no_diff_mae], axis=0)  # PHILIPP COMMENTED THIS OUT. THIS IS MOVED BELOW NOW.
new_data["target"] = new_data["target"].round(2)



# ============================================================
# REMOVE ORIGINAL ROWS THAT HAVE BEEN REPLACED BY AVERAGES
# using exact ionization pair, not only parent molecule key
# ============================================================

# canonicalize pair strings on both sides
together_average_pkas["avg_pair_key"] = (
    together_average_pkas["micropka_input_1"]
    .astype(str)
    .apply(canonicalize_micropka_pair)
)
no_diff_mae["avg_pair_key"] = (  # PHILIPP ADDED THIS!
    no_diff_mae["micropka input"]
    .astype(str)
    .apply(canonicalize_micropka_pair)
)
new_data["pair_key"] = (
    new_data["micropka input"]
    .astype(str)
    .apply(canonicalize_micropka_pair)
)
_tap_keys = set(zip(  # PHILIPP ADDED THIS
    together_average_pkas['prefix'].astype(str).str.strip(),
    together_average_pkas['micropka_input_1'].astype(str).apply(canonicalize_micropka_pair)
))
no_diff_mae = no_diff_mae[
    ~no_diff_mae.apply(
        lambda r: (str(r['prefix']).strip(),
                   canonicalize_micropka_pair(str(r['micropka input']))) in _tap_keys,
        axis=1
    )
].reset_index(drop=True)
avg_pair_keys = (  # PHILIPP ADDED THIS AND COMMENTED OUT THE OTHER avg_pair_keys.
    set(zip(together_average_pkas["prefix"].astype(str).str.strip(),
            together_average_pkas["avg_pair_key"]))
    | set(zip(no_diff_mae["prefix"].astype(str).str.strip(),
              no_diff_mae["avg_pair_key"]))
)
#avg_pair_keys = set(
#    zip(
#        together_average_pkas["prefix"].astype(str).str.strip(),
#        together_average_pkas["avg_pair_key"], no_diff_mae["avg_pair_key"]
#    )
#)

new_data["replace_key"] = list(
    zip(
        new_data["prefix"].astype(str).str.strip(),
        new_data["pair_key"]
    )
)
before = len(new_data)

new_data = new_data[
    ~new_data["replace_key"].isin(avg_pair_keys)
].copy()

after = len(new_data)

print("Rows removed by averaged exact microstate-pair replacement:", before - after)

new_data = new_data.drop(columns=["pair_key", "replace_key"])
new_data = pd.concat([new_data, no_diff_mae], axis=0, ignore_index=True)  # PHILIPP ADDED THIS

# (broader drop) remove any of the averages from the combined datasets + ml meets pKa 
together_average_pkas['canonical smiles'] = together_average_pkas['SMILES_1'].apply(canonicalize)
#together_average_pkas["key"] = together_average_pkas["canonical smiles"].map(mol_key)
#new_data["key"]  = new_data["canonical smiles"].map(mol_key)
#test_keys = set(together_average_pkas["key"].dropna())
#new_data = new_data[~new_data["key"].isin(test_keys)].copy()
#new_data= new_data.drop(columns=["key"])
# add the averages into the combined datasets + ml meets pKa
together_average_pkas_ = together_average_pkas.drop(['SMILES_1','SMILES_2','FP','micropka_input_2','deprot_1','deprot_2'], axis=1)
together_average_pkas_ = together_average_pkas_.rename(columns={'micropka_input_1':'micropka input','citations':'citation'})
together_average_pkas_['target'] = [i for i in together_average_pkas_['average']]
new_data['average'] = [i for i in new_data['target']]
new_data['pKas'] = [[i] for i in new_data['target']]
new_data['mean absolute error'] = [np.nan for i in range(len(new_data))]
new_data['Tanimoto'] = [np.nan for i in range(len(new_data))]
new_data['absolute difference'] = [np.nan for i in range(len(new_data))]
new_data = pd.concat([together_average_pkas_, new_data], ignore_index=True) 
new_data = mol_and_fp(new_data, 'canonical smiles')
print('Length of clean data without D2A-pKa comparison: ' + str(len(new_data)))

print('==================================================== COMPARING WITH D2A-PKA DATA ==========================================================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
# compare this new_data of averages, external datasets from epik paper, and ML meets pKa with D2A-pKa datan
nd_d2a_matches = find_micropka_tanimoto_overlaps(new_data, d2a_oxygen_, 1.00) 
nd_d2a_df = pd.DataFrame(nd_d2a_matches) # 690 overlapping; we will use these to confirm if our ionized sites are reliable 
# sanity check
all_match_ = nd_d2a_df['FP_prot_1'].eq(nd_d2a_df['FP_prot_2']).all()
print('Do all overlapping fingerprints match?: ' + str(all_match_))
print(str(len(nd_d2a_df)) + ' molecules overlapping with our dataset and D2A-pKa, which we will use to confirm if our ionized sites are reliable; the rest of D2A-pKa can go into the dataset for training')

print('================================================== CHECKING IONIZED STATES WITH D2A-PKA ===================================================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
diff_ionization = compare_micropka_inputs(nd_d2a_df)
# since find_micropka_tanimoto_overlaps is used, then there will be 0 mismatch in ionization
nd_d2a_matches_prefix = find_tanimoto_matches(new_data, d2a_oxygen_, 1.00) # use this since it just takes the prefix and the first molecule in prot_mol>>deprot_mol
nd_d2a_df_prefix = pd.DataFrame(nd_d2a_matches_prefix)
diff_ionization_ = compare_micropka_inputs(nd_d2a_df_prefix) # run compare_micropka_inputs again with the above overlaps ^
print('# of mismatching ionized sites from the ' + str(len(nd_d2a_df)) + ' overlapping molecules: ' + str(len(diff_ionization)))
print('Citations from our data that mismatch ionized sites with D2A-pKa: ' + str(diff_ionization['citation_1'].unique()))
print("Look at lines 817 to 819 of code --> there are 11 mismatches in ionization ")
print("From our data these are the ionizations: " + str(diff_ionization_['micropka_input_1'].values))
print("From D2A-pKa these are the ionizations: " + str(diff_ionization_['micropka_input_2'].values))
print('From the "Deprotonation Sites of Acetohydroxamic Acid Isomers. A Theroetical and Experimental Study." paper, we can conclude that O ionization is more favorable, so we will keep these 11 mismatches with our data ionization ')

# remove  overlapping ionizated molecules from d2a-pKa, so we can concatenate it with new_data
nd_d2a_df[['SMILES_1','deprot_1']] = nd_d2a_df['micropka_input_1'].str.split('>>',expand=True)
duplicate_keys = set(zip(nd_d2a_df['prefix'], nd_d2a_df['SMILES_1']))
mask = d2a_oxygen_[['prefix', 'canonical smiles']].apply(tuple, axis=1).isin(duplicate_keys)
d2a_oxygen_ = d2a_oxygen_[~mask].copy()
# separate D2A-pKa based on if there are averages in pKa, if not then it is reliable and we can train on
d2a_oxygen_['pka_vals'] = d2a_oxygen_.pka_vals.apply(lambda x: x.strip('[]').split(' ')) # turn pka_vals column from str type to list 
reliable_d2a = d2a_oxygen_[d2a_oxygen_['pka_vals'].apply(lambda lst: len(lst) == 1)]
# drop inagreeable ionization pairs
d2a_bad_ionization_pairs = set(diff_ionization_["micropka_input_2"].dropna().astype(str).apply(canonicalize_micropka_pair))
reliable_d2a["pair_key"] = (reliable_d2a["micropka input"].astype(str).apply(canonicalize_micropka_pair))
reliable_d2a = reliable_d2a[~reliable_d2a["pair_key"].isin(d2a_bad_ionization_pairs)].copy()
reliable_d2a = reliable_d2a.drop(columns=["pair_key"], errors="ignore")

print(str(len(reliable_d2a)) + ' pKas from D2A-pKa can be added to our trainable dataset')
unreliable_d2a = d2a_oxygen_[d2a_oxygen_['pka_vals'].apply(lambda lst: len(lst) > 1)] # do NOT add to data 

# add reliable_d2a to new_data 
add_d2a = input("Would you like to add the reliable molecules from D2A-pKa to the dataset? (Y/N): ").strip().upper() # ask user if they want to add D2A-pKa data
if add_d2a == 'Y':
    reliable_d2a = reliable_d2a.rename(columns={'pka_vals':'pKas','target':'average'})
    reliable_d2a['mean absolute error'] = [np.nan for i in range(len(reliable_d2a))]
    reliable_d2a['absolute difference'] = [np.nan for i in range(len(reliable_d2a))]
    reliable_d2a['Tanimoto'] = [np.nan for i in range(len(reliable_d2a))]
    reliable_d2a['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in reliable_d2a['protonated']]
    new_data_ = pd.concat([new_data, reliable_d2a], axis=0, ignore_index=True)
    print(str(len(new_data_)) + ' length of data BEFORE filtering out external test set molecules')
else:
    new_data_ = new_data.copy()
    print(str(len(new_data_)) + ' length of data BEFORE filtering out external test set molecules')

# filter out any molecules from test set 
print('================================== FILTERING STEP: removing any molecules seen in external test sets ====================================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
nov_s = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/Novartis/test.source', names=['reaction_smiles'])
nov_t = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/Novartis/test.target', names=['target'])
nov = pd.concat([nov_s, nov_t], axis=1)
lit_s = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/Literature/test.source', names=['reaction_smiles'])
lit_t = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/Literature/test.target', names=['target'])
lit = pd.concat([lit_s, lit_t], axis=1)
sam6_s = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL6/test.source', names=['reaction_smiles'])
sam6_t = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL6/test.target', names=['target'])
sam6 = pd.concat([sam6_s, sam6_t], axis=1)
sam6_taut_s = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL6_DEC_10_2025/test.source', names=['reaction_smiles'])
sam6_taut_t = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL6_DEC_10_2025/test.target', names=['target'])
sam6_taut = pd.concat([sam6_taut_s, sam6_taut_t], axis=1)
sam6_s_31 = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL6/31_molecules/test.source', names=['reaction_smiles'])
sam6_t_31 = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL6/31_molecules/test.target', names=['target'])
sam6_31 = pd.concat([sam6_s_31, sam6_t_31], axis=1)
sam7_s = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL7/test.source', names=['reaction_smiles'])
sam7_t = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL7/test.target', names=['target'])
sam7 = pd.concat([sam7_s, sam7_t], axis=1)
sam8_s = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL8/test.source', names=['reaction_smiles'])
sam8_t = pd.read_csv('/scratch/cii2002/pka/NEW_EXTERNAL_TEST/REGRESSION/prot_to_deprot_pairs/SAMPL8/test.target', names=['target'])
sam8 = pd.concat([sam8_s, sam8_t], axis=1)
external = pd.concat([nov, lit, sam6, sam6_taut, sam6_31, sam7, sam8], axis=0)
external['prefix'] = external['reaction_smiles'].apply(classify_reaction)
external[['protonated','deprotonated']] = external['reaction_smiles'].str.split('>>',expand=True)
external = mol_and_fp(external, 'protonated')
external['canonical smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in external['protonated']]

# ============================================================
# OPTION B: STRICT PARENT-MOLECULE FILTER
# Drop any training row whose parent molecule (uncharged, largest
# fragment) appears in ANY external test set, regardless of which
# specific ionization event the training row represents.
# ============================================================

breakpoint()

external = external.reset_index(drop=True)
new_data_ = new_data_.reset_index(drop=True)

def canonical_and_charge(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True)
    charge = Chem.GetFormalCharge(mol)
    return can, charge


def neutral_key_from_pair_side(prot_smiles, deprot_smiles, pka):
    prot_can, prot_charge = canonical_and_charge(prot_smiles)
    deprot_can, deprot_charge = canonical_and_charge(deprot_smiles)

    if pd.isna(pka):
        return None

    pka = round(float(pka), 6)

    keys = []

    if prot_charge == 0:
        keys.append((prot_can, pka))

    if deprot_charge == 0:
        keys.append((deprot_can, pka))

    return keys

external_keys = set()

for _, row in external.iterrows():
    keys = neutral_key_from_pair_side(
        row["protonated"],
        row["deprotonated"],
        row["target"]
    )

    if keys is not None:
        external_keys.update(keys)

def row_matches_external(row):
    keys = neutral_key_from_pair_side(
        row["prot_smiles"],
        row["deprot_smiles"],
        row["average"]
    )

    if keys is None:
        return False

    return any(k in external_keys for k in keys)


drop_mask = new_data_.apply(row_matches_external, axis=1)

rows_to_drop = new_data_[drop_mask].copy()
new_data_filtered = new_data_[~drop_mask].copy()

print("Original new_data_:", len(new_data_))
print("Rows dropped:", len(rows_to_drop))
print("Filtered new_data_:", len(new_data_filtered))

new_data_ = new_data_filtered.copy()
new_data_ = new_data_.reset_index(drop=True)


breakpoint()


def parent_inchikey(smi):
    """Return InChIKey of the standardized, uncharged parent molecule."""
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _lfc.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

# Build the test parent set from external (use the protonated half;
# parent_inchikey strips charges so either half gives the same key).
external["parent_key"] = external["protonated"].apply(parent_inchikey)
test_parent_keys = set(external["parent_key"].dropna())

# For training, derive parent from the protonated (left) half of micropka input
# for consistency. Either half works since parent_inchikey neutralizes.
new_data_["proton_half"] = new_data_["micropka input"].str.split(">>", n=1).str[0]
new_data_["parent_key"]  = new_data_["proton_half"].apply(parent_inchikey)

# Drop training rows whose parent appears in any test set
mask_drop = new_data_["parent_key"].isin(test_parent_keys)
df_filtered = new_data_[~mask_drop].copy()
df_filtered = df_filtered.drop(columns=["parent_key", "proton_half"])

print(f"Parent-molecule filter (Option B): "
      f"{mask_drop.sum()} training rows dropped because their parent appears in a test set "
      f"({len(test_parent_keys)} unique test parents).")

testing_df_for_now = df_filtered.copy()

breakpoint()  # — leaving in your existing breakpoint here if you want it

# Replace lines 1181–1185 with: OLD WAY OPTION A
#new_data_["pair_key"] = new_data_["micropka input"].apply(canonicalize_micropka_pair)
#external["pair_key"]  = external["reaction_smiles"].apply(canonicalize_micropka_pair)
#test_pair_keys = set(zip(external["prefix"].astype(str).str.strip(), external["pair_key"]))
#df_filtered = new_data_[
#    ~new_data_.apply(
#        lambda r: (str(r["prefix"]).strip(), r["pair_key"]) in test_pair_keys, axis=1)
#].copy()
#df_filtered = df_filtered.drop(columns=["pair_key"])
#print(f"InChIKey/pair filter removed {len(new_data_) - len(df_filtered)} training rows")

#new_data_["key"] = new_data_["canonical smiles"].map(mol_key)
#external["key"]  = external["canonical smiles"].map(mol_key)
#test_keys = set(external["key"].dropna())
#df_filtered = new_data_[~new_data_["key"].isin(test_keys)].copy()
#df_filtered = df_filtered.drop(columns=["key"])
#testing_df_for_now = df_filtered.copy()

#breakpoint()

# filter out based on tanitomo similarity
def mol_from_smiles(s):
    try:
        return Chem.MolFromSmiles(s)
    except:
        return None
def morgan_fp(mol, radius=2, nBits=2048, useChirality=USE_CHIRALITY):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=nBits, useChirality=useChirality
    )

# ============================================================
# DROP TRAINING MOLECULES TOO SIMILAR TO EXTERNAL SET
# ============================================================

# ----------------------------
# Standardization helpers
# ----------------------------

_lfc = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()

def parent_mol(smiles):
    """
    Standardize molecule before fingerprinting without tautomer enumeration.
    """
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _lfc.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Could not standardize SMILES: {smiles} | Error: {e}")
        return None


def mol_to_parent_smiles(smiles):
    """
    Return standardized parent canonical SMILES.
    Useful for debugging and duplicate checks.
    """
    mol = parent_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def morgan_fp(mol, radius=2, nBits=2048, useChirality=USE_CHIRALITY):
    """
    Generate Morgan fingerprint.
    """
    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nBits,
        useChirality=useChirality
    )


# ----------------------------
# Main overlap function
# ----------------------------

def tanimoto_overlap_check_all(
    external_df,
    train_df,
    ext_smiles_col="canonical smiles",
    train_smiles_col="canonical smiles",
    threshold=0.85,
    radius=2,
    nBits=2048,
    useChirality=USE_CHIRALITY,
    require_same_prefix=False,
    prefix_col="prefix"
):
    """
    Find ALL training rows with Tanimoto similarity >= threshold
    to ANY external molecule.

    This fixes the common bug where only the single best match is dropped.

    Parameters
    ----------
    external_df : pd.DataFrame
        External/test/benchmark dataframe.

    train_df : pd.DataFrame
        Training dataframe to filter.

    ext_smiles_col : str
        SMILES column in external_df.

    train_smiles_col : str
        SMILES column in train_df.

    threshold : float
        Tanimoto threshold for removing training molecules.

    useChirality : bool
        If True, stereochemistry affects fingerprints.
        If False, stereoisomers are treated more similarly.

    require_same_prefix : bool
        If True, only drop training rows when external and training prefixes match.
        If False, drop all acidic/basic rows for similar parent molecules.
        For avoiding external leakage, False is usually safer.

    Returns
    -------
    matches_df : pd.DataFrame
        Every external-training pair with similarity >= threshold.

    train_matches_df : pd.DataFrame
        Unique training rows that should be dropped.
    """

    # Keep train_index labels one-to-one with rows. Later .loc/.drop(index=...)
    # use these labels, so duplicate indexes would drop unrelated rows.
    train_df = train_df.reset_index(drop=True).copy()

    # ----------------------------
    # Build standardized training fingerprints
    # ----------------------------
    train_records = []

    for train_idx, row in train_df.iterrows():
        smi = row[train_smiles_col]
        mol = parent_mol(smi)

        if mol is None:
            continue

        fp = morgan_fp(
            mol,
            radius=radius,
            nBits=nBits,
            useChirality=useChirality
        )

        if fp is None:
            continue

        rec = {
            "train_index": train_idx,
            "train_smiles_original": smi,
            "train_parent_smiles": Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True
            ),
            "train_fp": fp,
        }

        if prefix_col in train_df.columns:
            rec["train_prefix"] = row[prefix_col]

        if "micropka input" in train_df.columns:
            rec["train_micropka_input"] = row["micropka input"]

        if "target" in train_df.columns:
            rec["train_target"] = row["target"]

        if "citation" in train_df.columns:
            rec["train_citation"] = row["citation"]

        train_records.append(rec)

    if len(train_records) == 0:
        print("No valid training fingerprints were generated.")
        return pd.DataFrame(), pd.DataFrame()

    train_fps = [r["train_fp"] for r in train_records]

    # ----------------------------
    # Compare every external molecule against all training molecules
    # ----------------------------
    matches = []

    for ext_idx, row in external_df.iterrows():
        ext_smi = row[ext_smiles_col]
        ext_mol = parent_mol(ext_smi)

        if ext_mol is None:
            continue

        ext_fp = morgan_fp(
            ext_mol,
            radius=radius,
            nBits=nBits,
            useChirality=useChirality
        )

        if ext_fp is None:
            continue

        sims = DataStructs.BulkTanimotoSimilarity(ext_fp, train_fps)

        ext_parent_smiles = Chem.MolToSmiles(
            ext_mol,
            canonical=True,
            isomericSmiles=True
        )

        ext_prefix = row[prefix_col] if prefix_col in external_df.columns else None

        for j, sim in enumerate(sims):
            if sim < threshold:
                continue

            train_rec = train_records[j]
            train_prefix = train_rec.get("train_prefix", None)

            if require_same_prefix:
                if ext_prefix is None or train_prefix is None:
                    continue
                if ext_prefix != train_prefix:
                    continue

            match = {
                "external_index": ext_idx,
                "train_index": train_rec["train_index"],
                "Tanimoto": float(sim),

                "external_smiles_original": ext_smi,
                "external_parent_smiles": ext_parent_smiles,
                "train_smiles_original": train_rec["train_smiles_original"],
                "train_parent_smiles": train_rec["train_parent_smiles"],

                "external_prefix": ext_prefix,
                "train_prefix": train_prefix,
            }

            if "micropka input" in external_df.columns:
                match["external_micropka_input"] = row["micropka input"]

            if "target" in external_df.columns:
                match["external_target"] = row["target"]

            if "citation" in external_df.columns:
                match["external_citation"] = row["citation"]

            if "train_micropka_input" in train_rec:
                match["train_micropka_input"] = train_rec["train_micropka_input"]

            if "train_target" in train_rec:
                match["train_target"] = train_rec["train_target"]

            if "train_citation" in train_rec:
                match["train_citation"] = train_rec["train_citation"]

            matches.append(match)

    matches_df = pd.DataFrame(matches)

    if len(matches_df) == 0:
        train_matches_df = train_df.iloc[0:0].copy()
    else:
        train_matches_df = train_df.loc[matches_df["train_index"].unique()].copy()

    return matches_df, train_matches_df


# ============================================================
# RUN FILTER
# ============================================================

print("============================================================")
print("CHECKING TRAINING / EXTERNAL TANIMOTO OVERLAP")
print("============================================================")

# Optional debugging columns
external = external.copy()
df_filtered = df_filtered.reset_index(drop=True).copy()

external["parent_smiles_for_tanimoto"] = external["canonical smiles"].apply(mol_to_parent_smiles)
df_filtered["parent_smiles_for_tanimoto"] = df_filtered["canonical smiles"].apply(mol_to_parent_smiles)

# Main overlap check
external_matches, pkachu_matches = tanimoto_overlap_check_all(
    external_df=external,
    train_df=df_filtered,
    ext_smiles_col="canonical smiles",
    train_smiles_col="canonical smiles",
    threshold=0.85,
    radius=2,
    nBits=2048,
    useChirality=USE_CHIRALITY,
    require_same_prefix=False
)

print("External-training match pairs with Tanimoto >= 0.85:", len(external_matches))
print("Unique training rows to drop:", len(pkachu_matches))

if len(external_matches) > 0:
    print("\nTop external-training overlaps:")
    print(
        external_matches
        .sort_values("Tanimoto", ascending=False)
        .head(20)[
            [
                "Tanimoto",
                "external_index",
                "train_index",
                "external_prefix",
                "train_prefix",
                "external_parent_smiles",
                "train_parent_smiles",
            ]
        ]
    )

# Save matched pairs for inspection
external_matches.to_csv(
    "external_train_tanimoto_matches_ge_0p95.csv",
    index=False
)

pkachu_matches.to_csv(
    "training_rows_dropped_due_to_external_tanimoto_ge_0p95.csv",
    index=True
)

# Actually drop from training
drop_idx = external_matches["train_index"].dropna().unique() if len(external_matches) > 0 else []

before_len = len(df_filtered)
df_filtered = df_filtered.drop(index=drop_idx).copy()
after_len = len(df_filtered)

print("\nTraining rows BEFORE Tanimoto filtering:", before_len)
print("Training rows AFTER Tanimoto filtering:", after_len)
print("Rows removed:", before_len - after_len)

# Reset index only after dropping
df_filtered = df_filtered.reset_index(drop=True)


# ============================================================
# VALIDATION: rerun check after dropping
# ============================================================

print("\n============================================================")
print("VALIDATING THAT NO TRAINING / EXTERNAL OVERLAPS REMAIN")
print("============================================================")

remaining_matches, remaining_train_matches = tanimoto_overlap_check_all(
    external_df=external,
    train_df=df_filtered,
    ext_smiles_col="canonical smiles",
    train_smiles_col="canonical smiles",
    threshold=0.85,
    radius=2,
    nBits=2048,
    useChirality=USE_CHIRALITY,
    require_same_prefix=False
)

print("Remaining external-training match pairs with Tanimoto >= 0.85:", len(remaining_matches))
print("Remaining unique training rows still overlapping:", len(remaining_train_matches))

if len(remaining_matches) > 0:
    print("\nWARNING: overlaps still remain. Showing top remaining matches:")
    print(
        remaining_matches
        .sort_values("Tanimoto", ascending=False)
        .head(20)[
            [
                "Tanimoto",
                "external_index",
                "train_index",
                "external_prefix",
                "train_prefix",
                "external_parent_smiles",
                "train_parent_smiles",
            ]
        ]
    )

    remaining_matches.to_csv(
        "WARNING_remaining_external_train_tanimoto_matches_ge_0p95.csv",
        index=False
    )
else:
    print("Success: no external-training Tanimoto overlaps >= 0.85 remain.")


# ============================================================
# OPTIONAL DUPLICATE CHECKS WITHIN FINAL TRAINING SET
# ============================================================

print("\n============================================================")
print("CHECKING DUPLICATES WITHIN FINAL TRAINING SET")
print("============================================================")

if "micropka input" in df_filtered.columns:
    print(
        "Duplicate micropka input rows:",
        df_filtered["micropka input"].duplicated(keep=False).sum()
    )

if "canonical smiles" in df_filtered.columns:
    print(
        "Duplicate canonical smiles rows:",
        df_filtered["canonical smiles"].duplicated(keep=False).sum()
    )

if {"prefix", "canonical smiles"}.issubset(df_filtered.columns):
    print(
        "Duplicate prefix + canonical smiles rows:",
        df_filtered.duplicated(
            subset=["prefix", "canonical smiles"],
            keep=False
        ).sum()
    )

if "parent_smiles_for_tanimoto" in df_filtered.columns:
    print(
        "Duplicate parent smiles rows:",
        df_filtered["parent_smiles_for_tanimoto"].duplicated(keep=False).sum()
    )

if {"prefix", "parent_smiles_for_tanimoto"}.issubset(df_filtered.columns):
    print(
        "Duplicate prefix + parent smiles rows:",
        df_filtered.duplicated(
            subset=["prefix", "parent_smiles_for_tanimoto"],
            keep=False
        ).sum()
    )

dupe_rows = df_filtered[
    df_filtered["micropka input"].duplicated(keep=False)
].sort_values(by=["micropka input"])

dupe_rows = df_filtered[df_filtered['micropka input'].duplicated(keep=False)].sort_values(by=['micropka input'])
if len(dupe_rows) == 0:
    print('Any presence of duplicates within the dataset was also removed as well as removal of molecules from external test sets')
else:
    print('there are still duplicates REDO')

print('Length of data AFTER filtering out external test set molecules: ' + str(len(df_filtered)))
print('===================================================== SPLITTING DATA: split 8:1:1 =======================================================')
print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
# Define which amino acids will be going into training and test set 
train_aa = ['Arg','Cys','Glu','Lys',                    # 3‑pKa amino acids ➔ train
            'Gln','Ala','Asn','Gly','Ile','Leu','Met']  # 2‑pKa ➔ train

test_aa  = ['Asp','His','Tyr',                    # 3‑pKa ➔ test
            'Phe','Pro','Ser','Thr','Trp','Val']  # 2‑pKa ➔ test

# add 3-letter code name to df_filtered
df = (df_filtered.merge(aa_df[['micropka input','Name']], on='micropka input',how='left'))
df['first'] = [x.split('>>')[0] for x in df['micropka input']]
df['second'] = [x.split('>>')[-1] for x in df['micropka input']]
# Keep only rows where average is between 2 and 12
df = df[(df['average'] >= 2) & (df['average'] <= 12)]
print('Adding 2-12 range, so length of data is: ' + str(len(df)))
# make sure all of the values in pKas column are floats not strings
df["pKas"] = df["pKas"].apply(
    lambda x: ([float(i) for i in x] if isinstance(x, list) else [float(i) for i in ast.literal_eval(x)] if isinstance(x, str) and x.startswith("[") else [float(x)] if isinstance(x, str) or isinstance(x, (int, float)) else None))
# change prefix labels from acidic and basic to Deprot and Prot
prefix = []
seq2seq = []
most_neutral = []          # holds the neutral SMILES with prefix like "Prot:<neutral>"
charges_first_col = []     # optional: to keep per-atom charges for inspection
charges_second_col = []

for i, (x, y) in enumerate(zip(df['first'], df['second'])):
    first_mol  = Chem.MolFromSmiles(x)
    second_mol = Chem.MolFromSmiles(y)

    if first_mol is None or second_mol is None:
        prefix.append('Invalid')
        seq2seq.append(f'Invalid:{x}>>{y}')
        most_neutral.append(f'Invalid:{x}')
        charges_first_col.append(None)
        charges_second_col.append(None)
        continue

    # per-atom formal charges
    charges_first  = [a.GetFormalCharge() for a in first_mol.GetAtoms()]
    charges_second = [a.GetFormalCharge() for a in second_mol.GetAtoms()]
    charges_first_col.append(charges_first)
    charges_second_col.append(charges_second)

    # count "charged atoms" (nonzero formal charge)
    n_charged_first  = sum(abs(c) != 0 for c in charges_first)
    n_charged_second = sum(abs(c) != 0 for c in charges_second)

    # tie-breaker: total absolute charge
    abs_sum_first  = sum(abs(c) for c in charges_first)
    abs_sum_second = sum(abs(c) for c in charges_second)

    # --- choose which is "more neutral" ---
    if (n_charged_first < n_charged_second) or (
        n_charged_first == n_charged_second and abs_sum_first < abs_sum_second
    ):
        neutral_smiles, other_smiles = x, y
        neutral_charges, other_charges = charges_first, charges_second
    elif (n_charged_second < n_charged_first) or (
        n_charged_first == n_charged_second and abs_sum_second < abs_sum_first
    ):
        neutral_smiles, other_smiles = y, x
        neutral_charges, other_charges = charges_second, charges_first
    else:
        # Still tied: keep original order as "neutral" heuristic
        neutral_smiles, other_smiles = x, y
        neutral_charges, other_charges = charges_first, charges_second

    # --- decide Prot vs Deprot from neutral → other ---
    total_neutral = sum(neutral_charges)
    total_other   = sum(other_charges)
    total_diff    = total_other - total_neutral

    if total_diff > 0:
        this_prefix = 'Prot'      # neutral → more positive overall
    elif total_diff < 0:
        this_prefix = 'Deprot'    # neutral → more negative overall
    else:
        # same total charge: look at per-atom deltas
        deltas = [b - a for a, b in zip(neutral_charges, other_charges)]
        pos_up   = sum(d > 0 for d in deltas)   # e.g., 0 → +1
        pos_down = sum(d < 0 for d in deltas)   # e.g., +1 → 0 or 0 → -1
        if pos_up > pos_down:
            this_prefix = 'Prot'
        elif pos_down > pos_up:
            this_prefix = 'Deprot'
        else:
            this_prefix = 'Neutral'

    prefix.append(this_prefix)
    most_neutral.append(f'{this_prefix}:{neutral_smiles}')
    seq2seq.append(f'{this_prefix}:{neutral_smiles}>>{other_smiles}')
df['charges_first']  = charges_first_col
df['charges_second'] = charges_second_col
df['prefix']   = prefix
df['seq2seq']  = seq2seq
df['most_neutral'] = most_neutral

# organize inputs for seq2seq from micropka input
df[["input for seq2seq", "target for seq2seq"]] = df["micropka input"].apply(lambda x: pd.Series(seq2seq_pair(x)))
df['new_column'] = df['prefix'].str.cat(df['input for seq2seq'], sep=':')
df['group_id'] = df['first'].apply(charge_invariant_key)
# reverse prefix for augmentation later
df['rev_prefix'] = df['prefix'].replace({'Prot': 'Deprot', 'Deprot': 'Prot'})

# split data 
train, val, test = group_random_split(df, group_col="group_id")
train_size = len(train)
val_size = len(val) 
test_size = len(test) 

# get averages; they will all be put into test set 
#average_groups   = set(df.loc[df['Tanimoto'].notna(), 'group_id'])
average_groups = set(
    df.loc[df['citation'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 1),
           'group_id']
)
train_aa_groups  = set(df.loc[df['Name'].isin(train_aa), 'group_id'])
test_aa_groups   = set(df.loc[df['Name'].isin(test_aa),  'group_id'])
#print('Number of averages in data: ' + str(len(average_groups)))
print('Number of multi-pKa groups forced into test: ' + str(len(average_groups)))

rng = np.random.default_rng(42)

# --- helper: rows in a set of groups ---
def rows_for_groups(group_ids):
    if not group_ids: return 0
    return len(df[df['group_id'].isin(group_ids)])

# --- locked sets ---
must_train = set(train_aa_groups)               # AA-for-train
must_test  = set(average_groups) | set(test_aa_groups)  # averages + AA-for-test
locked     = must_train | must_test

# --- feasibility check (hard stop if impossible) ---
lt, lv, ls = train_size, val_size, test_size
if rows_for_groups(must_train) > lt:
    raise AssertionError(
        f"Locked train rows ({rows_for_groups(must_train)}) exceed train target ({lt}). "
        "Increase train_size or relax must_train."
    )
if rows_for_groups(must_test)  > ls:
    raise AssertionError(
        f"Locked test rows ({rows_for_groups(must_test)}) exceed test target ({ls}). "
        "Increase test_size or relax must_test."
    )

# --- build a group order that’s stable+random but respects original split preference ---
# 1) base preference: groups that were originally in train, then val, then test (keeps changes minimal)
def groups_in(df_part): return list(df_part['group_id'].drop_duplicates())

base_train_groups = groups_in(train)
base_val_groups   = groups_in(val)
base_test_groups  = groups_in(test)

# Remove locked & dedupe while preserving order
def ordered_unique(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

pref_order = ordered_unique(
    [g for g in base_train_groups if g not in locked] +
    [g for g in base_val_groups   if g not in locked] +
    [g for g in base_test_groups  if g not in locked]
)

# Any remaining groups that weren’t in the base splits (should be none, but just in case)
all_groups = list(df['group_id'].drop_duplicates())
tail = [g for g in all_groups if g not in locked and g not in pref_order]
# Shuffle tail deterministically for fairness
rng.shuffle(tail)
pool_order = pref_order + tail

# --- start from empty splits and add locked groups first ---
def take_groups_from_list(order, taken_set, need_rows):
    picked = []
    total = 0
    for g in order:
        if g in taken_set or g in locked:  # skip already used or locked (locked are handled separately)
            continue
        g_rows = len(df[df['group_id'] == g])
        picked.append(g)
        total += g_rows
        if total >= need_rows:
            break
    return picked

# initialize assignments
assign = {}  # group_id -> {'train'|'val'|'test'}

for g in must_train: assign[g] = 'train'
for g in must_test:  assign[g] = 'test'

# current row tallies
def tally(label):
    gids = [g for g,a in assign.items() if a == label]
    return rows_for_groups(gids)

cur_train = tally('train')
cur_test  = tally('test')
cur_val   = tally('val')  # 0 initially

# --- fill TRAIN to target with non-locked groups ---
need_train = max(0, lt - cur_train)
picked = take_groups_from_list(pool_order, set(assign.keys()), need_train)
for g in picked: assign[g] = 'train'
cur_train = tally('train')

# --- fill VAL to target with non-locked groups ---
need_val = max(0, lv - cur_val)
picked = take_groups_from_list(pool_order, set(assign.keys()), need_val)
for g in picked: assign[g] = 'val'
cur_val = tally('val')

# --- everyone else goes to TEST ---
for g in all_groups:
    if g not in assign:
        assign[g] = 'test'
cur_test = tally('test')

# If test is over the target, trim *non-locked* groups from the end of pool_order
over = cur_test - ls
if over > 0:
    # build test group list in pool-order to drop earliest-added (least preferred)
    test_groups = [g for g in pool_order if assign.get(g) == 'test' and g not in must_test]
    drop = []
    dropped_rows = 0
    for g in reversed(test_groups):  # reverse -> drop the least preferred last-added first
        drop.append(g)
        dropped_rows += len(df[df['group_id']==g])
        if dropped_rows >= over:
            break
    for g in drop:
        assign.pop(g, None)  # unassign
    # if we unassigned anything, put them nowhere (they were only surplus). Recompute cur_test.
    cur_test = tally('test')

# --- final sanity (exact sizes) ---
assert tally('train') == lt, f"train={tally('train')} vs target {lt}"
assert tally('val')   == lv, f"val={tally('val')} vs target {lv}"
assert tally('test')  == ls, f"test={tally('test')} vs target {ls}"
assert tally('train') + tally('val') + tally('test') == len(df), "Row conservation failed"

# --- materialize DataFrames ---
def build_split(label):
    gids = [g for g,a in assign.items() if a == label]
    return df[df['group_id'].isin(gids)].copy().reset_index(drop=True)

train = build_split('train')
val   = build_split('val')
test  = build_split('test')

print('Lengeth of train|val|test: ' + str(len(train)) + '|' + str(len(val)) + '|' + str(len(test)))
print('Total data size: ' + str(len(df)))

# use train, val, test to do 5 kfold random splitting while still keeping molecule with more than one ionization together in train or val

def charge_invariant_key_from_micropka(src_str: str) -> str:
    """
    src_str is a micropKa input like:
        'Prot:SMI1>>SMI2'  or  'SMI1>>SMI2'
    We take the left side (SMI1), strip any 'Prot:'/'Deprot:' labels,
    neutralize/standardize, then canonicalize to get the group_id.
    """
    left = src_str.split('>>', 1)[0]
    # remove optional prefix label "Prot:" / "Deprot:" if present
    if ':' in left and left.split(':', 1)[0] in ('Prot', 'Deprot', 'basic', 'acidic'):
        left = left.split(':', 1)[1]
    m = _to_mol(left)
    if m is None:
        return f"BAD:{left}"  # stable fallback so all such rows still group together
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)

# ---------- kfold over groups (with shuffle) ----------
def kfold_groups_indices(groups: pd.Series, n_splits=5, seed=42):
    """
    Returns a list of (train_idx, val_idx) tuples where each fold holds out
    a disjoint subset of 'groups'. All rows with the same group stay together.
    The groups are shuffled once with 'seed' and chunked into n_splits folds.
    """
    rng = np.random.default_rng(seed)
    # unique groups in stable order (first appearance)
    uniq = groups.drop_duplicates().tolist()
    rng.shuffle(uniq)  # randomized kfold over groups

    # chunk groups
    folds_g = []
    size = int(np.ceil(len(uniq) / n_splits))
    for i in range(n_splits):
        folds_g.append(set(uniq[i*size : (i+1)*size]))

    idx = np.arange(len(groups))
    out = []
    for i in range(n_splits):
        val_mask = groups.isin(folds_g[i]).to_numpy()
        val_idx = idx[val_mask]
        train_idx = idx[~val_mask]
        out.append((train_idx, val_idx))
    return out

result = pd.concat([train, val], axis=0, ignore_index=True)
folds = kfold_groups_indices(result['group_id'], n_splits=5, seed=42)


# save files 
while True:
    question = input("Do you want to save the split data? (Y/N): ").strip().upper()
    if question == 'Y':
        path = input('Input path of where you want the files to be saved: ').strip()
        # ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # save full dataset
        df.to_csv(os.path.join(path, 'full_pKa_CHU.csv'), index=False)

        # save kfold random splitting (unchanged—only micropka/average here)
        for i in range(1,6):
            os.makedirs(os.path.join(path, str(i)), exist_ok=True)
        for i, (tr_idx, va_idx) in enumerate(folds, start=1):
            train_folds = result.iloc[tr_idx].copy()
            val_folds   = result.iloc[va_idx].copy()
            fold_dir = os.path.join(path, str(i))
            train_folds['micropka input'].to_csv(os.path.join(fold_dir, 'train.source'), index=False, header=False)
            train_folds['average'].to_csv(       os.path.join(fold_dir, 'train.target'), index=False, header=False)
            val_folds['micropka input'].to_csv(  os.path.join(fold_dir, 'val.source'),   index=False, header=False)
            val_folds['average'].to_csv(         os.path.join(fold_dir, 'val.target'),   index=False, header=False)

        # save train split
        train['rev_seq2seq_input'] = train['rev_prefix'].str.cat(train['target for seq2seq'], sep=':')
        train['micropka input'].to_csv(os.path.join(path, 'train.source'), header=False, index=False)
        train['average'].to_csv( os.path.join(path, 'train.target'), header=False, index=False)
        train['new_column'].to_csv(os.path.join(path, 'train_seq2seq.source'), header=False, index=False)
        train['target for seq2seq'].to_csv(os.path.join(path, 'train_seq2seq.target'), header=False, index=False)
        train[['micropka input','average','citation']].to_csv(os.path.join(path, 'train.csv'), index=False)
        train['rev_seq2seq_input'].to_csv(os.path.join(path, 'train_seq2seq_augmented.source'), header=False, index=False)
        train['input for seq2seq'].to_csv(os.path.join(path, 'train_seq2seq_augmented.target'), header=False, index=False)

        # save val split
        val['rev_seq2seq_input'] = val['rev_prefix'].str.cat(val['target for seq2seq'], sep=':')
        val['micropka input'].to_csv(os.path.join(path, 'val.source'), header=False, index=False)
        val['average'].to_csv( os.path.join(path, 'val.target'), header=False, index=False)
        val['new_column'].to_csv(os.path.join(path, 'val_seq2seq.source'), header=False, index=False)
        val['target for seq2seq'].to_csv(os.path.join(path, 'val_seq2seq.target'), header=False, index=False)
        val[['micropka input','average','citation']].to_csv(os.path.join(path, 'val.csv'), index=False)
        val['rev_seq2seq_input'].to_csv(os.path.join(path, 'val_seq2seq_augmented.source'), header=False, index=False)
        val['input for seq2seq'].to_csv(os.path.join(path, 'val_seq2seq_augmented.target'), header=False, index=False)

        # save test split
        test['rev_seq2seq_input'] = test['rev_prefix'].str.cat(test['target for seq2seq'], sep=':')
        test['micropka input'].to_csv(os.path.join(path, 'test.source'), header=False, index=False)
        test['average'].to_csv( os.path.join(path, 'test.target'), header=False, index=False)
        test['new_column'].to_csv(os.path.join(path, 'test_seq2seq.source'), header=False, index=False)
        test['target for seq2seq'].to_csv(os.path.join(path, 'test_seq2seq.target'), header=False, index=False)
        test[['micropka input','pKas','average','citation','mean absolute error']].to_csv(os.path.join(path, 'test.csv'), index=False)
        test['rev_seq2seq_input'].to_csv(os.path.join(path, 'test_seq2seq_augmented.source'), header=False, index=False)
        test['input for seq2seq'].to_csv(os.path.join(path, 'test_seq2seq_augmented.target'), header=False, index=False)

        print(f"All files written to {path}")
        break

    elif question == 'N':
        print("Skipping save.")
        break

    else:
        print("Please type 'Y' or 'N'.")  

breakpoint()
print('END OF SCRIPT')
