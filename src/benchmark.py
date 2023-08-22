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


def main():
    start = time.time()
    dh.ensure_dir_exists("results")
    subj_result_file = dh.prepare_output_file()
    train = dh.get_or_extract_features(config.subjects, config.rootdir)
    test = dh.get_or_extract_features(config.heldout_subjects, config.heldout_dir, train_feats=False)
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
