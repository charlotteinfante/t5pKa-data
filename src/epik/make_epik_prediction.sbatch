#!/bin/bash
#SBATCH --job-name=pka 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --time=3:00:00
#SBATCH --mem=20GB 
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=cii2002@nyu.edu

cd /scratch/cii2002/MolGpKa-data/src/epik/

module purge
module load schrodinger/2021-1

$SCHRODINGER/ligprep -s 1 -t 1 -i 0 -ismi novartis.smi -omae novartis.mae.gz -WAIT
$SCHRODINGER/epik -scan -imae novartis.mae.gz -omae novartis_pka.mae.gz -ph 7.4 -p 0.1 -highest_pka 14.0 -lowest_pka 0.0 -WAIT
$SCHRODINGER/utilities/sdconvert -imae novartis_pka.mae.gz -osdf novartis_pka.sdf.gz -annstereo -pKa
