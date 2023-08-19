"""
---------------------------------------------------------------
Perform and Evaluate Classification on the Heldout Set
---------------------------------------------------------------

Description:
    This script is designed to perform and evaluate classification 
    on a specified heldout set of data. 

Usage:
    This code was used to create the baseline in the corresponding
    research paper. It should only be used with data splits where 
    true labels are available.

Note:
    This script is part of the research project and is provided 
    for reproducibility purposes.

"""


from datetime import timedelta
import numpy as np
import os
import time

import extract_features as fe
import classifier
import config
import data_helpers as dh

def ensure_dir_exists(directory):
    """
    Ensure that the specified directory exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def get_or_extract_features(subjects, dir, train_feats=True):
    print("Extracting features")
    """
    Extract features for all subjects.
    """
    features = {}
    for subject in subjects:
        print(f"Extracting features for subject {subject}")
        for feature_set in config.feature_sets:
            features.setdefault(feature_set, {})
            ensure_dir_exists("features")
            feature_path = os.path.join("features", f"{subject}_{feature_set}.npy")
            if os.path.isfile(feature_path):
                print("Loading saved features: ", feature_set)
                features[feature_set].update(np.load(feature_path,allow_pickle='TRUE').item())
            else:
                print("Extracting features: ", feature_set)
                if train_feats:
                    f_nr = dh.read_mat_file(os.path.join(dir, f"results{subject}_NR.mat"))
                    f_tsr = dh.read_mat_file(os.path.join(dir, f"results{subject}_TSR.mat"))
                    fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
                    fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")
                else:
                    f = dh.read_mat_file(os.path.join(dir, f"results{subject}.mat"))
                    fe.extract_sentence_features(subject, f, feature_set, features, "")
                only_subjects = {subj_feat: features[feature_set][subj_feat] for subj_feat in features[feature_set] if subj_feat.startswith(subject)}
                np.save(feature_path, only_subjects)

    if config.plot_all_subjects_features: 
        dh.plot_all_subjects_feature_distribution(features)
    subjects_feats, labels, idxs, full_idxs = {}, {}, {}, {}
    
    for feature_set in config.feature_sets:
        subjects_feats[feature_set] = {}
        labels[feature_set] = {}
        idxs[feature_set] = {}
        full_idxs[feature_set] = {}

        for feat, values in features[feature_set].items():
            parts = feat.split("_")
            subj, idx, full_idx = parts[0], parts[-2], parts[-1]

            subjects_feats[feature_set].setdefault(subj, []).append(values[:-1])
            labels[feature_set].setdefault(subj, []).append(values[-1])
            idxs[feature_set].setdefault(subj, []).append(idx)
            full_idxs[feature_set].setdefault(subj, []).append(full_idx)
    
    return {'features': subjects_feats, 'labels': labels, 'full_idxs': full_idxs, 'idxs': idxs} 

def main():
    start = time.time()
    if not os.path.exists("results"):
         os.makedirs("results")
    subj_result_file = dh.prepare_output_file()
    train = get_or_extract_features(config.subjects, config.rootdir)
    test = get_or_extract_features(config.heldout_subjects, config.heldout_dir, train_feats=False)
    if config.pca_preprocessing:
        train, test = dh.apply_pca_preprocessing(train, test)
    for feats in train['features']:
        logs = dh.prepare_logs()
        for _ in range(config.runs):
            results = classifier.benchmark(train['features'][feats], \
                train['labels'][feats], test['features'][feats], test['labels'][feats])
            logs = dh.update_logs(logs, results)
        dh.log_results(logs, feats, subj_result_file)
 
    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()
