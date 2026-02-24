# Generate pKa Dataset for Model Training
### Generate Calculated pKa data using Schrödinger's Epik version 2021
Using `src/epik/ACD_CHEMBL_pka_short.smi` as an **example** smiles file

1. Go to src/epik and run make_epik_prediction.sbatch to get epik predictions
```bash
sbatch make_epik_prediction.sbatch
```

Output is epik_predicts.csv, and this file will give us the atom number needed to be feed into ionize_data.py

2. Run src/ionize_data.py, which reads in epik_predicts.csv and it will spit out the necessary files for training
```bash
python ionize_data.py
```

Make sure that ionize_data.py is reading in the correct file (check out last three lines in the file)

### Generate Calculated pKa data WITHOUT Schrödinger's Epik
If you do not have access to Epik, then we made it avaliable (in collaberation with MolGpKa) to use the predict_protonate.py script without needing to use Epik. This script allows you to use any other program of your choice as long as you have the atom number and pKa information as well as it allows you to soley use MolGpKa's predictions, and saves the predicted information into files ready to be read by T5Chem-pKa. MolGpKa is trained on Epik's predictions [
](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00075), so results should be similar as to using Epik. **We strongly recommend using Epik version 2021, if you are able to.**

Option:

A. Using only SMILES files (with no information on pKa, BasicOrAcid, and atom number)
```bash
python ionize.py --data datasets/train.source --save /path/to/be/saved/in/
```

B. Using a file with information containing 'acd_pKa','BasicOrAcid', and 'Atom' number
```bash
python ionize.py --data src/datasets/CHEMBL_EX_USING_molgpka_atomnum.csv --save /path/to/be/saved/in/
```

Remember to clean data --> remove any duplicate molecules 
