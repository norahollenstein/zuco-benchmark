import config_topoplots as config
import h5py
import extract_features as fe
import numpy as np

for subject in config.subjects:
    subj_results = []
    print(subject)
    filename_nr = config.rootdir + "results" + subject + "_NR.mat"
    filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

    f_nr = h5py.File(filename_nr, 'r')
    sentence_data_nr = f_nr['sentenceData']

    f_tsr = h5py.File(filename_tsr, 'r')
    sentence_data_tsr = f_tsr['sentenceData']

    features_nr = {}
    features_tsr = {}

    for feature_set in config.feature_sets:
        features_nr[feature_set] = {}
        features_tsr[feature_set] = {}

        fe.extract_sentence_features(subject, f_nr, sentence_data_nr, feature_set, features_nr, "NR")
        fe.extract_sentence_features(subject, f_tsr, sentence_data_tsr, feature_set, features_tsr, "TSR")

        print(features_nr)

        nr_feat_vec = []
        for sample_id, features in features_nr[feature_set].items():
            nr_feat_vec.append(features[:-1])

        print(len(nr_feat_vec))

        print(np.mean(nr_feat_vec, axis=0))

        tsr_feat_vec = []
        for sample_id, features in features_tsr[feature_set].items():
            tsr_feat_vec.append(features[:-1])

        print(len(tsr_feat_vec))

        print(np.mean(tsr_feat_vec, axis=0))

