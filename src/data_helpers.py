from datetime import datetime
from pathlib import Path
import h5py
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import json
import config


def prepare_output_file():
    """Create all output files required for reporting results"""
    result_file = open("results/" + str(
        datetime.now()) + "_svm_results_" + config.class_task + "_" + config.dataset + "_random" + str(
        config.randomized) + "_" + config.kernel + ".csv", "a")
    return result_file


def read_mat_file(filename):
    """Read MATLAB files with EEG data"""
    mat_file = h5py.File(filename, 'r')
    sentence_data = mat_file['sentenceData']
    return sentence_data


def plot_feature_distribution(subj, dataset, feature_dict, feature_set):
    """Plot feature distribution for a single feature"""
    colors = ["#44A2C4", "#B3D882"]
    feature_file = open("feature-plots/"+feature_set+"-"+dataset+".csv", "a")
    # get plots for largest elements
    data = pd.DataFrame(columns=["subject", "feat", "label"])
    for i, (x, y) in enumerate(feature_dict.items()):
        data.loc[i] = [x[:3], np.mean(y[0:len(y)-1]), y[-1].split("_")[0]]
    # print("HEAD ", data.head(10))
    fig, ax = plt.subplots()
    # print(subj, np.mean(data['feat']), np.std(data['feat']), np.min(data['feat']), np.max(data['feat']), file=feature_file)
    ax = sns.violinplot(x="subject", y="feat", hue="label",
                        data=data, palette=colors)  # , inner="stick")
    try:
        ax.collections[0].set_edgecolor("#337F9A")  # "#337F9A"
        ax.collections[1].set_edgecolor("#337F9A")  # "#337F9A"
        ax.collections[2].set_edgecolor("#92D050")
        ax.collections[3].set_edgecolor("#92D050")
    except IndexError:
        pass
    ax.set_title(feature_set)
    ax.set(xticklabels=[])
    ax.set(xticks=[])
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.savefig("feature_plots/" + feature_set + "_" + subj+".pdf")
    plt.close()


def plot_all_subjects_feature_distribution(feature_dict):
    """Plot feature distributions for a single feature for all subjects"""
    colors = ["#44A2C4", "#B3D882"]
    plt.figure(figsize=(15, 8))
    Path("feature-plots/all_features_all_subjects").mkdir(parents=True, exist_ok=True)
    for feature_set in feature_dict:
        # get plots for largest elements
        data = pd.DataFrame(columns=["Feature", "Label"])
        for i, (x, y) in enumerate(feature_dict[feature_set].items()):
            data.loc[i] = [np.mean(y[0:len(y)-1]), y[-1].split("_")[0]]
        rc = {'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 10,
              'axes.titlesize': 20, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
        sns.set(rc=rc)
        sns.set_style("whitegrid", {'axes.grid': False})
        fig, ax = plt.subplots()
        ax = sns.violinplot(x="Label", y="Feature", dodge=False, data=data,
                            palette=colors, inner='quartile')  # , inner="stick")
        yposlist = data.groupby(['Label'])['Feature'].median().tolist()
        positions = data.groupby(['Label'])['Feature'].quantile(.55).tolist()
        xposlist = range(len(yposlist))
        for i in range(len(yposlist)):
            ax.text(xposlist[i]-0.125, positions[i],
                    "m = "+str(round(yposlist[i], 1)))
        ax.set_title(feature_set)
        ax.set_xlabel('')
        ax.set_ylabel('')
        fig.savefig("feature-plots/all_features_all_subjects/"+ feature_set + "_all_subjects.pdf")
        plt.close()


def plot_explained_variance(feature_set, explained_variance_ratios):
    """ Plot the amont of variance explained by the PCA-features for each subject """
    NUM_COLORS = len(config.subjects)
    Path("feature-plots/elbow/").mkdir(parents=True, exist_ok=True)
    color = cm.rainbow(np.linspace(0, 1, NUM_COLORS))
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111)
    for index, (subject, c) in enumerate(zip(config.subjects, color)):
        print(index, subject, c)
        ax.plot(
            np.cumsum(explained_variance_ratios[index]), label=subject, c=c, linewidth=3)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40)
    plt.xlabel('Number of Components', fontsize=50)
    plt.ylabel('Cumulative Explained Variance', fontsize=50)
    plt.title("Explained Variance for " + feature_set, fontsize=50)
    plt.legend(fontsize=40, ncol=2)
    plt.savefig("feature-plots/elbow/" + feature_set + ".pdf")


def plot_electrode_weights_pca(subjects_feats, feature_set, mode):
    """ Plot influence of individual electrodes on PCA-features """
    pca = PCA(n_components=0.95)
    Path("feature-plots/electrode_weights_pca/").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(25, 15))
    subjects_data = []
    for subj in subjects_feats[feature_set]:
        scaling = MinMaxScaler(feature_range=(0, 1)).fit(
            subjects_feats[feature_set][subj])
        scaled = scaling.transform(subjects_feats[feature_set][subj])
        pca.fit_transform(scaled)
        componenets = abs(pca.components_)
        var_ratios = pca.explained_variance_ratio_
        assert componenets.shape[0] == var_ratios.shape[0]
        # apply realtive importance of each component ?
        weighted_ratio = var_ratios.reshape(-1, 1) * componenets
        assert weighted_ratio.shape == componenets.shape
        ratios_sum = np.sum(weighted_ratio, axis=0)
        subjects_data.append(ratios_sum)
    subjects_data = np.array(subjects_data)
    print('subjects_data.shape: ', subjects_data.shape)
    rc = {'font.size': 20, 'axes.labelsize': 25, 'legend.fontsize': 20,
          'axes.titlesize': 30, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
    sns.set(rc=rc)
    sns.set_style("whitegrid", {'axes.grid': False})
    ax = sns.heatmap(data=subjects_data)
    ax.set_title(
        "Relative Electrode Importance for PCA Features: " + feature_set)
    subjects_labels = [subj for subj in subjects_feats[feature_set]]
    ax.set(yticklabels=subjects_labels)
    ax.set_xlabel('Electrodes')
    ax.set_ylabel('Subjects')
    import csv
    # You will need 'wb' mode in Python 2.x
    with open("feature-plots/electrode_weights_pca/electrode_weights_pca_" + feature_set+"_"+mode + ".csv", 'w') as f:
        w = csv.writer(f)
        w.writerows(subjects_data)
    plt.savefig("feature-plots/electrode_weights_pca/" +
                feature_set+"_"+mode + ".pdf")


def pca_preprocessing(features, pca_n_components, mode='train'):
    "fit and apply pca to subjects"
    subjects_feats = features['features']
    labels = features['labels']
    print('pca preprocessing')
    pca = PCA(n_components=pca_n_components)
    for feature_set in subjects_feats:
        if config.plot_electrode_weights_pca:
            print("Plotting plot_electrode_weights_pca")
            plot_electrode_weights_pca(subjects_feats, feature_set, mode)
        print("using pca pre-processing")
        for subj in subjects_feats[feature_set]:
            # scale before applying pca
            scaling = MinMaxScaler(feature_range=(0, 1)).fit(
                subjects_feats[feature_set][subj])
            subjects_feats[feature_set][subj] = scaling.transform(
                subjects_feats[feature_set][subj])
            # fit pca
            pca.fit(subjects_feats[feature_set][subj])
            subjects_feats[feature_set][subj] = pca.transform(
                subjects_feats[feature_set][subj])
            print('feat ', feature_set, " subject: ", subj, "Explained_variance_ratio sum : ", np.sum(
                pca.explained_variance_ratio_*100))
    return {'features': subjects_feats, 'labels': labels}


def determine_pca_n_components(features):
    """
    Determine the number of components to use for PCA.
    """
    subjects_feats = features['features']
    subjects_pca_results = []
    for feature_set in subjects_feats:
        subjects_explained_variance = []
        for subj in subjects_feats[feature_set]:
            pca = PCA(n_components=config.explained_variance)
            pca.fit(subjects_feats[feature_set][subj])
            subjects_pca_results.append(len(pca.components_))
            subjects_explained_variance.append(pca.explained_variance_ratio_)
            print('feat ', feature_set, " subject: ", subj,
                  "Explained_variance_ratio sum : ", pca.explained_variance_ratio_)
        if config.plot_explained_variance:
            plot_explained_variance(feature_set, subjects_explained_variance)
    print('n components ', np.median(subjects_pca_results))
    # print('n components ', int(math.ceil(np.median(subjects_pca_results))))
    return int(math.ceil(np.median(subjects_pca_results)))


def apply_pca_preprocessing(train, test):
    """ Apply PCA preprocessing only to some parts of the data"""
    eeg_features_train = {'features': {}, 'labels': {}}
    et_features_train = {'features': {}, 'labels': {}}
    eeg_features_test = {'features': {}, 'labels': {}}
    et_features_test = {'features': {}, 'labels': {}}
    contains_electrode_features = False
    for feature_set in config.feature_sets:
        if 'electrode' in feature_set:
            contains_electrode_features = True
            eeg_features_train['features'][feature_set] = train['features'][feature_set]
            eeg_features_train['labels'][feature_set] = train['labels'][feature_set]

            eeg_features_test['features'][feature_set] = test['features'][feature_set]
            eeg_features_test['labels'][feature_set] = test['labels'][feature_set]
        else:
            et_features_train['features'][feature_set] = train['features'][feature_set]
            et_features_train['labels'][feature_set] = train['labels'][feature_set]

            et_features_test['features'][feature_set] = test['features'][feature_set]
            et_features_test['labels'][feature_set] = test['labels'][feature_set]

    if contains_electrode_features:
        pca_n_components = determine_pca_n_components(eeg_features_train)
        print("Using ", pca_n_components, " components for PCA")
        eeg_features_train = pca_preprocessing(
            eeg_features_train, pca_n_components, mode='train')
        eeg_features_test = pca_preprocessing(
            eeg_features_test, pca_n_components, mode='test')
        train['features'] = {**eeg_features_train['features'], **et_features_train['features']}
        train['labels'] = {**eeg_features_train['labels'], **et_features_train['labels']}
        test['features'] =  {**eeg_features_test['features'], **et_features_test['features']}
        test['labels'] = {**eeg_features_test['labels'], **et_features_test['labels']}
    return train, test


def prepare_logs():
    "Prepare log files in case the labels are also available"
    # preds, test_y, acc, p,r, f1, train_acc, bacc, bf1, bp, br, boot_preds
    n_subj = len(config.heldout_subjects)
    accuracies = [[] for _ in range(n_subj)]
    predictions = [[] for _ in range(n_subj)]
    f1s = [[] for _ in range(n_subj)]
    ps = [[] for _ in range(n_subj)]
    rs = [[] for _ in range(n_subj)]
    baccs = [[] for _ in range(n_subj)]
    bf1s = [[] for _ in range(n_subj)]
    bps = [[] for _ in range(n_subj)]
    brs = [[] for _ in range(n_subj)]
    boot_preds_all = [[] for _ in range(n_subj)]
    return [[predictions, accuracies, f1s, ps, rs],
            [boot_preds_all, baccs, bf1s, bps, brs]]


def update_logs(logs, results):
    "Save log files in case the labels are also available"
    n_subj = len(config.heldout_subjects)
    normal, bootstrap = logs[:]
    normal_res, bootstrap_res = results[:]
    for n, r in zip(normal, normal_res):
        for i_subj in range(n_subj):
            n[i_subj].append(r[i_subj])
    if config.bootstrap:
        for bn, br in zip(bootstrap, bootstrap_res):
            for i_subj in range(n_subj):
                bn[i_subj].extend(br[i_subj])
    return [normal] + [bootstrap]


def log_results(logs, feats, subj_result_file):
    "Save log files in case the labels are also available"
    normal, bootstrap = logs
    for index, subject in enumerate(config.heldout_subjects):
        # print results for individual subjects to file
        if config.bootstrap:
            Path("bootstrapping_low").mkdir(parents=True, exist_ok=True)
            with open('bootstrapping_low/bootstrpping_acc_'+subject+'_'+feats+'.npy', 'wb') as f:
                np.save(f, np.array(bootstrap[1][index]))
            with open('bootstrapping_low/bootstrapping_f1_'+subject+'_'+feats+'.npy', 'wb') as f:
                np.save(f, np.array(bootstrap[2][index]))
            with open('bootstrapping_low/bootstrapping_p_'+subject+'_'+feats+'.npy', 'wb') as f:
                np.save(f, np.array(bootstrap[3][index]))
            with open('bootstrapping_low/bootstrapping_r_'+subject+'_'+feats+'.npy', 'wb') as f:
                np.save(f, np.array(bootstrap[4][index]))
            with open('bootstrapping_low/bootstrapping_preds_'+subject+'_'+feats+'.npy', 'wb') as f:
                np.save(f, np.array(bootstrap[0][index]))

        print("Classification test accuracy : ",  subject, feats,
              np.mean(normal[1][index]), np.std(normal[1][index]))
        print(subject, feats, 'accuracy',  np.mean(normal[1][index]), np.std(
            normal[1][index]), file=subj_result_file)

        print("Classification test f1 : ",  subject, feats,
              np.mean(normal[2][index]), np.std(normal[2][index]))
        print(subject, feats, 'f1', np.mean(normal[2][index]), np.std(
            normal[2][index]), file=subj_result_file)

        print("Classification test precision : ",  subject,
              feats, np.mean([index]), np.std(normal[3][index]))
        print(subject, feats, 'precision', np.mean(normal[3][index]), np.std(
            normal[3][index]), file=subj_result_file)

        print("Classification test recall : ",  subject, feats,
              np.mean(normal[4][index]), np.std(normal[4][index]))
        print(subject, feats, 'recall', np.mean(normal[4][index]), np.std(
            normal[4][index]), file=subj_result_file)

        print("Saving subject ", subject, " to predictions file")
        with open('predictions/'+subject+'_'+feats+'.npy', 'wb') as f:
            np.save(f, np.array(normal[0][index]))

 ################### No Labels are available ##############################


def prepare_logs_predictions():
    "Prepare log files for predictions when no labels are available"
    n_subj = len(config.heldout_subjects)
    predictions = [[] for _ in range(n_subj)]
    return predictions


def update_logs_predictions(logs, results):
    "Update log files for predictions when no labels are available"
    print("Updating logs")
    n_subj = len(config.heldout_subjects)
    for i_subj in range(n_subj):
        logs[i_subj].append(results[i_subj])
    return logs


def write_logs_predictions(logs, feats):
    "Save log files for predictions when no labels are available"
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    for index, subject in enumerate(config.heldout_subjects):
        print("Saving subject ", subject, " to predictions file")
        with open('predictions/'+subject+'_'+feats+'.npy', 'wb') as f:
            np.save(f, np.array(logs[index]))


def create_submission(logs, feature, idxs, model="svm"):
    "Create Submission for benchmark"
    directory = "submissions/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("Creating submission for feature: ", feature)
    submission_file = {}
    for index, subject in enumerate(config.heldout_subjects):
        submission_file[subject] = {}
        # If we have multiple runs, we just select the first
        result = logs[index][0].tolist()
        idx = idxs[feature][subject]
        for pred, id in zip(result, idx):
            submission_file[subject][id] = pred
    with open(f'{directory}{datetime.now().strftime("%Y%m%d-%H%M%S")}_{model}_{feature}.json', 'w') as f:
        json.dump(submission_file, f)