# Generate pKa Dataset for Model Training
### Generate Calculated pKa data using Schrondinger's Epik 
Use ACD_CHEMBL_pka_short.smi as an example smiles file

Go to src/epik and run make_epik_prediction.sbatch to get epik predictions

Output should be epik_predicts.csv, and this file will give us the atom number needed to be feed into protonation script
