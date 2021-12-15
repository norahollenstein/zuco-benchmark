
# dataset {zuco1, zuco2, zucoAll, zuco1sr, zuco1sr_only}
dataset="zuco2"
heldout_subjects = ["XBB", "XDT", "XLS", "XPB", "XSE", "XTR", "XWS", "XAH", "XBD", "XSS"]
heldout_dir = "../../methlab/ETH_AS/SAMUEL/Holdout/FirstLevelV2_concat_unfold_correctlyMergedSacc_avgref/"
subjects = [ 'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH,  - YRH, YMS
rootdir = "../../mehtlab_loc/ETH_AS/FirstLevelV2_concat_unfold_correctlyMergedSacc_avgref/"


# experiment setup
seed = 1
runs = 1
plot_top_electrodes = False
plot_all_subjects_features = False
# experiment modifications
mean_electrodes = False
pca_preprocessing = True
bootstrap = False
# why wouldn't you want to log results? 
log_results = True
# median amount of variance to be explaiend by PCA
explained_variance = 0.95
plot_explained_variance = False
plot_electrode_weights_pca=False
# randomize labels as a sanity check; default = False
randomized_labels = False

# sentence-level eye-tracking feature sets
#feature_sets = ["fixation_number", "omission_rate", "reading_speed", 'sent_gaze', "mean_sacc_dur", "max_sacc_velocity", "mean_sacc_velocity", "max_sacc_dur", "max_sacc_amp", "mean_sacc_amp", 'sent_saccade', 'sent_gaze_sacc', "theta_mean", "alpha_mean", "beta_mean", "gamma_mean", "eeg_means", "sent_gaze_eeg_means",  "electrode_features_theta", "electrode_features_alpha", "electrode_features_beta", "electrode_features_gamma", "electrode_features_all", "electrode_gaze_sacc"]
#feature_sets = ["reading_speed"]

# sentence-level baseline feature
feature_sets = ["flesch_baseline"]
#electrode features 
#feature_sets = ["electrode_features_gamma", "electrode_features_alpha", "electrode_features_beta", "electrode_features_all", "electrode_features_theta"]

# feature sets for benchmark 
feature_sets = ["electrode_features_all", "sent_gaze_sacc", "sent_gaze_sacc_eeg_means"]

# classification task {tasks, sessions, subjects, tasks-cross-subj, blocks, blocks-in-sets}
class_task = 'tasks-cross-subj'


# SVM params
kernel = 'linear' # only linear kernel allows for analysis of coefficients

# EEG information
chanlocs = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20',
            'E22',
            'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
            'E40',
            'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58',
            'E59',
            'E60', 'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77',
            'E78',
            'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96',
            'E97',
            'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112',
            'E114',
            'E115', 'E116', 'E117', 'E118', 'E120', 'E121', 'E122', 'E123', 'E124']

