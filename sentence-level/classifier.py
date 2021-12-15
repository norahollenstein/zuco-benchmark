from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import config

def build_data(data, labels):
    """
    Builds the data for the classification task.
    TODO vectorize
    """
    X, y = [],[]
    for subj in data:
        for it in data[subj]:
            X.append(it)
        for label in labels[subj]:
            if label == "NR":
                y.append(1)
            else:
                y.append(0)

    return X, y


def bootstrap_confidence(X, y, clf, n_bootstraps=100, seed_value=0):
    from sklearn.utils import resample
    accs, ps, rs, f1s, b_preds = [], [], [], [], []
    for i in range(n_bootstraps):
        resampled_X, resampled_y = resample(X, y, replace=True, n_samples=len(X), random_state=seed_value+i)
        prediction = clf.predict(resampled_X)
        accuracy = len([i for i, j in zip(prediction, resampled_y) if i == j]) / len(resampled_y)
        p,r,f1,_ = precision_recall_fscore_support(resampled_y, prediction, average='macro')
        accs.append(accuracy)
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        b_preds.append(prediction)
    return accs, f1s, ps, rs, b_preds

def benchmark(trainX, trainy, testX, testy, seed_value, randomized=False, bootstrap=False): 
    """Classification for the benchmark task."""

    np.random.seed(seed_value)
    import time
    start = time.process_time()
    #preprocess data
    print("\nPre-processing ")
    if randomized:
        print("Warning: Random labels")

    train_X, train_y = build_data(trainX, trainy)
    test_X, test_y, subjs= [], [], []
    for subj in testX:
        subj_X, subj_y = build_data({subj:testX[subj]}, {subj:testy[subj]})
        #subj_X, subj_y = shuffle(subj_X, subj_y)
        subjs.append(subj)
        test_X.append(subj_X)
        test_y.append(subj_y)
    
    train_X, train_y = shuffle(train_X, train_y)

    # scale feature values
    scaling = MinMaxScaler(feature_range=(0, 1)).fit(train_X)
    train_X = scaling.transform(train_X)
    test_X = [scaling.transform(subj) for subj in test_X]
    
    print("\nTraining on all subjects in the train split ")
    # train SVM classifier
    clf = SVC(random_state=seed_value, kernel=config.kernel, gamma='scale', cache_size=1000, probability=False)
    clf.fit(train_X, train_y)
    #clfs = bootstrap_confidence_training(train_X, train_y, n_bootstraps=100, seed_value=seed_value, plot=True)
    train_acc = len([i for i, j in zip(clf.predict(train_X), train_y) if i == j]) / len(train_y)
    print("train acc ", train_acc)
    predictions, accuracies, f1s, ps,rs, baccs, bf1s, bps, brs, boot_preds, test_ys = [],[],[],[],[],[],[],[], [], [], []
    for index, subj in enumerate(subjs):
        print("\nPredicting on subject ", subj)
        #for clf in clfs:
        prediction = clf.predict(test_X[index])
        if randomized: 
            prop = 390/739
            prediction = np.random.choice([0,1], len(prediction), p=[prop, 1-prop]) 
        accuracy = len([i for i, j in zip(prediction, test_y[index]) if i == j]) / len(test_y[index])
        print('acc ', accuracy)
        #f1 = f1_score(test_y[index], prediction)
        p,r,f1,_ = precision_recall_fscore_support(test_y[index], prediction, average='macro')
        # TODO improve logging
        print('f1 ', f1)
        predictions.append(prediction)
        accuracies.append(accuracy)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        test_ys.append(test_y[index])
        if bootstrap:
            print("Performing bootstrapping")
            bacc, bf1, bp, br, b_preds = bootstrap_confidence(test_X[index], test_y[index], clf, n_bootstraps=100, seed_value=seed_value)
            baccs.append(bacc)
            bf1s.append(bf1)
            bps.append(bp)
            brs.append(br)
            boot_preds.append(b_preds)
    return predictions, test_ys, accuracies, ps, rs, f1s, train_acc, baccs, bf1s, bps, brs,  boot_preds