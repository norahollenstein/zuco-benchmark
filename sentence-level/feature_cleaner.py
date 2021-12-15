import numpy as np
import os
from sklearn.metrics import classification_report


import config
import data_helpers as dh

def feature_cleaner(subjects):
    features = {}
    for subject in subjects:
        print("Extracting features for subject {}".format(subject))
        #filename_nr = dir + "results" + subject + "_NR.mat"
        #filename_tsr = dir + "results" + subject + "_TSR.mat"
        for feature_set in config.feature_sets:
            if feature_set not in features:
                features[feature_set] = {}
            if not os.path.exists("features"): 
                os.makedirs("features")
            feature_path_old = os.path.join("old_features", subject+"_"+feature_set+".npy")
            feature_path_new = os.path.join("features", subject+"_"+feature_set+".npy")

            if os.path.isfile(feature_path_old):
                features[feature_set] = np.load(feature_path_old,allow_pickle='TRUE').item()
                print("loading")
                features[feature_set][subject] = {}
                for feat in features[feature_set]:
                    if feat.split("_")[0] == subject:
                        features[feature_set][subject] = {feat:features[feature_set][feat]}
                #print(features[feature_set].keys())
                np.save(feature_path_new, features[feature_set][subject])


def main():
    feature_cleaner(config.subjects)
    feature_cleaner(config.heldout_subjects)


if __name__ == "__main__":
    main()