# Using DNA language models for the IBIS Challenge
This repository contains scripts for obtaining a solution for the IBIS Challenge (2024) using DNA language models DNABERT-2, GENA-LM and Nucleotide Transformer (NT).

Scripts for GENA-LM and Nucleotide Transformer are run through Jupyter Notebook, for DNABERT-2 are run through command-line.
## Setup environment
### For GENA-LM and NT
```
conda create -n env python=3.12.2
conda activate env
python3 -m pip install -r requirements.txt
```
### For DNABERT-2
```
conda create -n dna python=3.8.20
conda activate dna
cd DNABERT-2
python3 -m pip install -r requirements.txt
```
## Data
Data packages used in the challenge are available at the link: https://zenodo.org/records/15056803 We used data of Final examples.

For datasets assembling you should use the bibis package (https://github.com/autosome-ru/ibis-challenge)

Before fine-tuning, make sure that current folder contains the folder 'test' with HTS, SMS and PBM data files in fasta format.
## Fine-tuning
### For GENA-LM and NT
Make sure that current folder contains the folder 'datasets' with the HTS and CHS+gHTS data files in parquet.gzip format. All files should represent a dataframe with columns `seq, group, label` in the 1st row and DNA sequence, group (train or val) and class label in other rows.

For each discipline (A2G and G2A) all transcription factors are used steadily during fine-tuning.
### For DNABERT-2
The scripts for DNABERT-2 fine-tuning is based on the original scripts available at the link (https://github.com/MAGICS-LAB/DNABERT_2) and are distributed in accordance with the license.

Before fine-tuning, make sure that for every transcription factor you have 3 csv files of your dataset: train.csv, dev.csv, and test.csv. The model is trained on train.csv and is evaluated on the dev.csv file; after the training, the best model file is loaded and be evaluated on test.csv. Files dev.csv and test.csv may be the same. All csv files with data should have head named `sequence, label` in the 1st row and DNA sequence and class label in other rows.

Every transcription factor (TF) should be used for fine-tuning individually.
```
cd DNABERT-2
export DATA_PATH=$path/to/data/folder  # folder with csv datasets of current TF
export disc=  # `a2g` or `g2a` discipline
export TF=  # name of TF from relevant discipline
python train_test_${disc}.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name ${TF} \
    --model_max_length 500 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --fp16 \
    --save_steps 200 \
    --output_dir ${TF}_out \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False
```
