"""
---------------------------------------------------------------
Perform Classification on the Heldout Set
---------------------------------------------------------------

Description:
    This script is designed to perform classification on a 
    specified heldout set of data.

Usage:
    This code can be used for a first baseline in the benchmarking 
    task. It is intended to serve as a starting point for 
    comparison with other models and configurations.

Note:
    Please ensure that you are using appropriate data splits in config.py

"""

from datetime import timedelta
import time

import classifier
import config
import data_helpers as dh


def main():
    start = time.time()
    dh.ensure_dir_exists("results")
    train = dh.get_or_extract_features(config.subjects, config.rootdir)
    test = dh.get_or_extract_features(config.heldout_subjects, config.heldout_dir, train_feats=False)
    
    if config.pca_preprocessing:
        train, test = dh.apply_pca_preprocessing(train, test)
        
    for feats in train['features']:
        logs = dh.prepare_logs_predictions()
        print(f"Using features: {feats}")
        
        for _ in range(config.runs):
            results = classifier.benchmark_baseline(train['features'][feats], \
                train['labels'][feats], test['features'][feats], test['labels'][feats])
            logs = dh.update_logs_predictions(logs, results)

        if config.save_prediction_npy:
            dh.write_logs_predictions(logs, feats)

        # Create Submission file for the benchmark
        if config.create_submission:
            dh.create_submission(logs, feats, test['idxs'])

    elapsed = (time.time() - start)
    print(str(timedelta(seconds=elapsed)))


if __name__ == '__main__':
    main()
