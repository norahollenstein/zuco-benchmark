from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import time

import extract_features as fe
import classifier
import config
import data_helpers as dh


"""
Perform classification on the heldout set for the benchmark task.
This is the code used to create the baseline in the paper. 

"""

def get_or_extract_features(subjects, dir):
    print("extracting features")
    """
    Extract features for all subjects.
    """
    features = {}
    for subject in subjects:
        print("Extracting features for subject {}".format(subject))
        filename_nr = dir + "results" + subject + "_NR.mat"
        filename_tsr = dir + "results" + subject + "_TSR.mat"
        for feature_set in config.feature_sets:
            if feature_set not in features:
                features[feature_set] = {}
            if not os.path.exists("features"): 
                os.makedirs("features")
            feature_path = os.path.join("features", subject+"_"+feature_set+".npy")
            if os.path.isfile(feature_path):
                features[feature_set] = np.load(feature_path,allow_pickle='TRUE').item()
                print("loading")
            else: 
                f_nr = dh.read_mat_file(filename_nr)
                f_tsr = dh.read_mat_file(filename_tsr)
                fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
                fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")
                np.save(feature_path, features[feature_set])

            print('sanity ', len(features[feature_set]))
    if config.plot_all_subjects_features: 
        dh.plot_all_subjects_feature_distribution(config.subjects, config.dataset, features)
    subjects_feats = {}
    labels = {}
    for feature_set in config.feature_sets: 
        subjects_feats[feature_set] = {}
        labels[feature_set] = {}
        for feat in features[feature_set]:
            subj = feat.split("_")[0]
            if not subj in subjects_feats[feature_set]: 
                subjects_feats[feature_set][subj] = []
                labels[feature_set][subj] = []
            subjects_feats[feature_set][subj].append(features[feature_set][feat][:-1])
            labels[feature_set][subj].append(features[feature_set][feat][-1])         
    return {'features': subjects_feats, 'labels': labels}

def main():
    start = time.time()
    subj_result_file = dh.prepare_output_file()
    train = get_or_extract_features(config.subjects, config.rootdir)
    test = get_or_extract_features(config.heldout_subjects, config.heldout_dir)
    if config.pca_preprocessing:
        train, test = dh.apply_pca_preprocessing(train, test)
    for feats in train['features']:
        if config.log_results:
            logs = dh.prepare_logs()
        for i in range(config.runs):
            #preds, test_y, acc, coefs = classifier.svm_cross_subj(feats, config.seed+i, subject, config.randomized_labels)
            results = classifier.benchmark(train['features'][feats], \
                train['labels'][feats], test['features'][feats], test['labels'][feats])
            if config.log_results:
                logs = dh.update_logs(logs, results)
        #print("Classification train accuracy clf: ",feats, np.mean(train_accuracies), np.std(train_accuracies))
        if config.log_results: 
            dh.write_logs(logs, feats, subj_result_file)

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()
