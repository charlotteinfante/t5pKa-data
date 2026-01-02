#!/bin/bash
#SBATCH --job-name=pka 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --time=3:00:00
#SBATCH --mem=20GB 
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=cii2002@nyu.edu

module purge
module load schrodinger/2021-1

$SCHRODINGER/ligprep -s 1 -t 1 -i 0 -ismi ACD_CHEMBL_pka_short.smi -omae ACD_CHEMBL_pka_short.mae.gz -WAIT
$SCHRODINGER/epik -scan -imae ACD_CHEMBL_pka_short.mae.gz -omae ACD_CHEMBL_pka_short_pka.mae.gz -ph 7.4 -p 0.1 -highest_pka 14.0 -lowest_pka 0.0 -WAIT
$SCHRODINGER/utilities/sdconvert -imae ACD_CHEMBL_pka_short_pka.mae.gz -osdf ACD_CHEMBL_pka_short_pka.sdf.gz -annstereo -pKa
python convert_sdf_to_csv --sdf ACD_CHEMBL_pka_short_pka.sdf.gz --original full_ACD_CHEMBL_pka_short.csv
