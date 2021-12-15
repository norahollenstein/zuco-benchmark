# Reading Task Classification

## Data

From ZuCo 1.0 and 2.0: NR and TSR tasks

On UZH server:  
ZuCo 1.0: methlab/NLP/Ce-ETH/FirstLevel_concat_unfold_correctlyMergedSacc_   
ZuCo 2.0: methlab/NLP/Ce_ETH/2019/FirstLevelV2_concat_unfold_correctlyMergedSacc

On spaceML:  
noraho@spaceml3:/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_preprocessed_sep2020  
noraho@spaceml3:/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco2_preprocessed_sep2020


## Classification with sentence-level features

Set parameters in config.py

Feature extraction:  
extract_features.py  
data_loading_helpers.py

Main script:  
classify_nr_trs.py  
Train & test classifier for each subject individually

classify_nr_tsr_cross.py  
Leave-one-out cross-subject models: train on all-1 subjects, test on left out subject

classify_sessions.py  
Classify recording sessions (this uses SR data from ZuCo 1)

classify_blocks.py  
Classify recording blocks from ZuCO 2

classify_subects.py  
Subject classification

classify_nr_trs_WordFixOnly.py  
Features include only data during fixation.  
Description: Section 6.3 Fixation ablation - Figure 20



## Classification with word-level features

Set parameters in config.py

Feature extraction:  
data_loading_helpers.py  
eeg_extractor.py  
gaze_extractor.py  
Best to do feature extraction once separately and save features for faster processing

tune_eeg_model_single.py  
tune_gaze_model_single.py  
Train & test classifier for each subject individually

tune_gaze_model_cross.py  
tune_eeg_model_cross.py  
Leave-one-out cross-subject models: train on all-1 subjects, test on left out subject
