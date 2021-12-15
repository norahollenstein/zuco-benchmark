import numpy as np
import extract_features as fe
import config
import data_helpers as dh
import time
from datetime import timedelta
import pandas as pd
import mne
from mne import EvokedArray
import matplotlib.pyplot as plt


# topoplots for NR, TSR, and NR-TSR across all sentences of all subjects


def main():

    start = time.time()

    features_all = pd.DataFrame(columns=['subj', 'feature_set', 'sample_id', 'feature_values', 'label'])

    for subject in config.subjects:
        print(subject)
        filename_nr = config.rootdir + "results" + subject + "_NR.mat"
        filename_tsr = config.rootdir + "results" + subject + "_TSR.mat"

        f_nr = dh.read_mat_file(filename_nr)
        f_tsr = dh.read_mat_file(filename_tsr)

        if config.dataset is "zuco1_sr":  # include sentiment reading as NR
            filename_sr = config.rootdir + "results" + subject + "_SR.mat"
            f_sr = dh.read_mat_file(filename_sr)

        features = {}

        for feature_set in config.feature_sets:

            features[feature_set] = {}

            fe.extract_sentence_features(subject, f_nr, feature_set, features, "NR")
            fe.extract_sentence_features(subject, f_tsr, feature_set, features, "TSR")
            if config.dataset is "zuco1_sr":
                fe.extract_sentence_features(subject, f_sr, feature_set, features, "NR")
            print(len(features[feature_set]), " samples collected for", feature_set)

            for x, y in features[feature_set].items():
                features_all = features_all.append({'subj': subject, 'feature_set': feature_set, 'sample_id': x, 'feature_values': np.array(y[:-1]), 'label':y[-1]}, ignore_index=True)

    for feature_set in config.feature_sets:

        info = mne.create_info(ch_names=config.chanlocs, ch_types="eeg", sfreq=500)

        features_nr = features_all.loc[(features_all['feature_set'] == feature_set) & (features_all['label'] == 'NR')]
        mean_nr = features_nr['feature_values'].mean()

        features_tsr = features_all.loc[(features_all['feature_set'] == feature_set) & (features_all['label'] == 'TSR')]
        mean_tsr = features_tsr['feature_values'].mean()

        diff = mean_nr - mean_tsr

        # NR
        evoked_nr = EvokedArray(mean_nr.reshape(-1,1), info=info)
        evoked_nr.set_montage("GSN-HydroCel-128")
        print(evoked_nr.info)
        evoked_nr.drop_channels([0])
        print(evoked_nr.info)

        fig, ax = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        ax = evoked_nr.plot_topomap(title='EEG patterns', time_unit='s', units='a.u.', scalings=1, vmin=min(diff), cmap='RdBu')
        plt.savefig("NR-topo-AVG-ALL"+feature_set+".pdf")
        plt.close()

        """

        # TSR
        evoked_tsr = EvokedArray(mean_tsr.reshape(-1, 1), info=info)
        evoked_tsr.set_montage("GSN-HydroCel-128")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        ax = evoked_tsr.plot_topomap(title='EEG patterns', time_unit='s', units='a.u.', scalings=1, vmin=min(diff), cmap='RdBu')
        plt.savefig("TSR-topo-AVG-ALL"+feature_set+".pdf")
        plt.close()

        # DIFF
        evoked_diff = EvokedArray(diff.reshape(-1, 1), info=info)
        evoked_diff.set_montage("GSN-HydroCel-128")

        fig, ax = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        ax = evoked_diff.plot_topomap(title='EEG patterns', time_unit='s', units='a.u.', scalings=1, vmin=min(diff), cmap='RdBu')
        plt.savefig("Diff-topo-AVG-ALL"+feature_set+".pdf")
        plt.close()
        """

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()
