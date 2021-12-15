
# dataset {zuco1, zuco2, zucoAll, zuco1_sr}
dataset = 'zuco2'

if dataset is 'zuco2':
    subjects = ['YAC']#, 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']  # exclude YMH
    rootdir = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"
elif dataset is 'zuco1' or dataset is 'zuco1_sr':
    subjects = ["ZJS", "ZDN", "ZJN", "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM"]
    rootdir = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"
elif dataset is "zucoAll":
    subjects = ["ZJS", "ZDN", "ZJN", "ZPH", "ZAB", "ZJM", "ZKB", "ZKH", "ZMG", "ZGW", "ZKW", "ZDM", 'YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS', 'YRH', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
    rootdir2 = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"
    rootdir1 = "/Volumes/methlab/NLP/Ce_ETH/OSF-ZuCo1.0-200107/mat7.3/"


# level {word, sentence}
# sentence-level: SVM used for classification
# word-level: LSTM used for classification
level = 'sentence'

seed = 1

# randomize labels as a sanity check; default = False
randomized_labels = False

feature_sets = ["electrode_features_gamma"]

# classification task {tasks, sessions, subjects}
class_task = 'tasks'

# classifier {svm, lstm}
classifier = 'svm'