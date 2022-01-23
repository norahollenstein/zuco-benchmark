import warnings
import data_loading_helpers as dlh
import numpy as np
import readability
from nltk import word_tokenize
import config


def flesch_reading_ease(text):
    """get Flesch reading ease score for a sentence."""

    tokenized = " ".join(word_tokenize(text))
    results = readability.getmeasures(tokenized, lang='en')
    fre = results['readability grades']['FleschReadingEase']

    return fre


def relabel_sessions(idx, label_orig):
    """split SR samples into session 1 and session 2"""

    if label_orig == "SR-Sess":
        if idx < 250:
            label = "Sess1"
        else:
            label = "Sess2"
    else:
        label = label_orig

    return label


def relabel_blocks(idx, label_orig):
    """Label sentence according to which experiment block they were recorded in"""

    nr1 = list(range(0, 50))
    nr2 = list(range(50, 100))
    nr3 = list(range(100, 151))
    nr4 = list(range(151, 201))
    nr5 = list(range(201, 251))
    nr6 = list(range(251, 300))
    nr7 = list(range(300, 349))
    tsr1 = list(range(0, 45))
    tsr2 = list(range(45, 117))
    tsr3 = list(range(117, 171))
    tsr4 = list(range(171, 236))
    tsr5 = list(range(236, 290))
    tsr6 = list(range(290, 350))
    tsr7 = list(range(350, 390))

    label = label_orig
    if label_orig == "NR":
        if idx in nr1:
            label = "NR_block1"
        elif idx in nr2:
            label = "NR_block2"
        elif idx in nr3:
            label = "NR_block3"
        elif idx in nr4:
            label = "NR_block4"
        elif idx in nr5:
            label = "NR_block5"
        elif idx in nr6:
            label = "NR_block6"
        elif idx in nr7:
            label = "NR_block7"
        else:
            print("UNKNOWN NR INDEX?!")
    elif label_orig == "TSR":
        if idx in tsr1:
            label = "TSR_block1"
        elif idx in tsr2:
            label = "TSR_block2"
        elif idx in tsr3:
            label = "TSR_block3"
        elif idx in tsr4:
            label = "TSR_block4"
        elif idx in tsr5:
            label = "TSR_block5"
        elif idx in tsr6:
            label = "TSR_block6"
        elif idx in tsr7:
            label = "TSR_block7"
        else:
            print("UNKNOWN TSR INDEX?!")


    return label


def extract_sentence_features(subject, f, feature_set, feature_dict, label_orig):
    """extract sentence level features from Matlab struct"""
    
        
    rawData = f['rawData']
    if label_orig!="":
        contentData = f['content']

    print(len(rawData))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for idx, sent_data in enumerate(rawData):

            label = label_orig
            full_idx = len(feature_dict[feature_set])

            if config.class_task == "sessions":
                label = relabel_sessions(idx, label)
            if config.class_task == "blocks" or config.class_task == "blocks-in-sets"\
                or config.class_task == "tasks_blocks":
                label = relabel_blocks(idx, label)

            if label_orig!="":
                obj_reference_content = contentData[idx][0]
                sent = dlh.load_matlab_string(f[obj_reference_content])

                # Flesch reading ease score
                fre = flesch_reading_ease(sent)

            # omission rate
            omissionR = f['omissionRate']
            obj_reference_omr = omissionR[idx][0]
            omr = np.array(f[obj_reference_omr])[0][0]

            # fixation number
            allFix = f['allFixations']
            obj_reference_allFix = allFix[idx][0]
            af = f[obj_reference_allFix]

            # mean saccade amplitude
            saccMeanAmp = f['saccMeanAmp']
            obj_reference_saccMeanAmp = saccMeanAmp[idx][0]
            smeana = np.array(f[obj_reference_saccMeanAmp])[0][0]
            smeana = smeana if not np.isnan(smeana) else 0.0


            # mean saccade duration
            saccMeanDur = f['saccMeanDur']
            obj_reference_saccMeanDur = saccMeanDur[idx][0]
            smeand = np.array(f[obj_reference_saccMeanDur])[0][0]
            smeand = smeand if not np.isnan(smeand) else 0.0

            # mean saccade velocity
            saccMeanVel = f['saccMeanVel']
            obj_reference_saccMeanVel = saccMeanVel[idx][0]
            smeanv = np.array(f[obj_reference_saccMeanVel])[0][0]
            smeanv = smeanv if not np.isnan(smeanv) else 0.0

            # saccade max amplitude
            saccMaxAmp = f['saccMaxAmp']
            obj_reference_saccMaxAmp = saccMaxAmp[idx][0]
            try:
                smaxa = np.array(f[obj_reference_saccMaxAmp])[0][0]
                smaxa = smaxa if not np.isnan(smaxa) else 0.0
            except IndexError:
                smaxa = 0.0

            # saccade max velocity
            saccMaxVel = f['saccMaxVel']
            obj_reference_saccMaxVel = saccMaxVel[idx][0]
            try:
                smaxv = np.array(f[obj_reference_saccMaxVel])[0][0]
                smaxv = smaxv if not np.isnan(smaxv) else 0.0
            except IndexError:
                smaxv = 0.0

            # saccade max duration
            saccMaxDur = f['saccMaxDur']
            obj_reference_saccMaxDur = saccMaxDur[idx][0]
            try:
                smaxd = np.array(f[obj_reference_saccMaxDur])[0][0]
                smaxd = smaxd if not np.isnan(smaxd) else 0.0
            except IndexError:
                smaxd = 0.0

            # EEG means
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                theta1 = f['mean_t1']
                obj_reference_t1 = theta1[idx][0]
                theta2 = f['mean_t2']
                obj_reference_t2 = theta2[idx][0]
                t_electrodes = np.nanmean(np.array([np.array(f[obj_reference_t1])[:105],np.array(f[obj_reference_t2])[:105]]), axis=0)
                t_mean = np.nanmean(t_electrodes)

                alpha1 = f['mean_a1']
                obj_reference_a1 = alpha1[idx][0]
                alpha2 = f['mean_a2']
                obj_reference_a2 = alpha2[idx][0]
                a_electrodes = np.nanmean(
                    np.array([np.array(f[obj_reference_a1])[:105], np.array(f[obj_reference_a2])[:105]]), axis=0)
                a_mean = np.nanmean(a_electrodes)

                beta1 = f['mean_b1']
                obj_reference_b1 = beta1[idx][0]
                beta2 = f['mean_b2']
                obj_reference_b2 = beta2[idx][0]
                b_electrodes = np.nanmean(
                    np.array([np.array(f[obj_reference_b1])[:105], np.array(f[obj_reference_b2])[:105]]), axis=0)
                b_mean = np.nanmean(b_electrodes)

                gamma1 = f['mean_g1']
                obj_reference_g1 = gamma1[idx][0]
                gamma2 = f['mean_g2']
                obj_reference_g2 = gamma2[idx][0]
                g_electrodes = np.nanmean(np.array([np.array(f[obj_reference_g1])[:105],np.array(f[obj_reference_g2])[:105]]), axis=0)
                g_mean = np.nanmean(g_electrodes)

            ### --- Text difficulty baseline --- ###
            if feature_set == "flesch_baseline":
                feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [fre, label]

            ### --- Sentencel-level eye tracking features --- ###
            if feature_set == "omission_rate":
                if not np.isnan(omr).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [omr, label]

            elif feature_set == "fixation_number":
                if 'duration' in af:
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [weighted_nFix, label]

            elif feature_set == "reading_speed":
                if 'duration' in af:
                    # convert sample to seconds
                    weighted_speed = (np.sum(np.array(af['duration']))*2/100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [weighted_speed, label]

            elif feature_set == "mean_sacc_dur":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smeand, label]

            elif feature_set == "mean_sacc_amp":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smeana, label]

            elif feature_set == "max_sacc_velocity":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smaxv, label]

            elif feature_set == "mean_sacc_velocity":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smeanv, label]

            elif feature_set == "max_sacc_dur":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smaxd, label]

            elif feature_set == "max_sacc_amp":
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smaxa, label]

            elif feature_set == "sent_gaze":
                if 'duration' in af:
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [omr, weighted_nFix, weighted_speed, smeand, label]

            elif feature_set == "sent_gaze_sacc":
                if 'duration' in af:
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [omr, weighted_nFix, weighted_speed, smeand, smaxv, smeanv, smaxd, smeana, smaxa, label]

            elif feature_set == "sent_saccade":
                if 'duration' in af:
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [smeand, smaxv, smeanv, smaxd, smeana, smaxa, label] 

            ### --- Sentencel-level EEG features --- ###
            elif feature_set == "theta_mean":
                if not np.isnan(t_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [t_mean, label]

            elif feature_set == "alpha_mean":
                if not np.isnan(a_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [a_mean, label]

            elif feature_set == "beta_mean":
                if not np.isnan(b_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [b_mean, label]

            elif feature_set == "gamma_mean":
                if not np.isnan(g_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [g_mean, label]

            elif feature_set == "eeg_means":
                if not np.isnan(g_mean) and not np.isnan(t_mean) and not np.isnan(b_mean) and not np.isnan(a_mean):
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [t_mean, a_mean, b_mean, g_mean, label]

            elif feature_set == "electrode_features_theta":
                if not np.isnan(t_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = np.concatenate(t_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_alpha":
                if not np.isnan(a_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = np.concatenate(a_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_beta":
                if not np.isnan(b_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = np.concatenate(b_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_gamma":
                if not np.isnan(g_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = np.concatenate(g_electrodes).ravel().tolist() + [label]

            elif feature_set == "electrode_features_all":
                if not np.isnan(g_electrodes).any() and not np.isnan(a_electrodes).any() and not np.isnan(t_electrodes).any() and not np.isnan(b_electrodes).any():
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = np.concatenate(t_electrodes).ravel().tolist() + np.concatenate(a_electrodes).ravel().tolist() + np.concatenate(b_electrodes).ravel().tolist() + np.concatenate(g_electrodes).ravel().tolist() + [label]

            elif feature_set == "sent_gaze_eeg_means":
                if 'duration' in af and not np.isnan(g_mean) and not np.isnan(t_mean) and not np.isnan(b_mean) and not np.isnan(a_mean):
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [omr, weighted_nFix, weighted_speed,
                                                                                         smeand, smaxv, smeanv, smaxd, t_mean, a_mean, b_mean, g_mean,
                                                                                         label]
                                                                                                                                                                  
            elif feature_set == "sent_gaze_sacc_eeg_means":
                if 'duration' in af and not np.isnan(g_mean) and not np.isnan(t_mean) and not np.isnan(b_mean) and not np.isnan(a_mean):
                    weighted_nFix = np.array(af['duration']).shape[0] / len(sent.split())
                    weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / len(sent.split())
                    feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = [omr, weighted_nFix, weighted_speed,
                                                                                         smeand, smaxv, smeanv, smaxd, smeana, smaxa, t_mean, a_mean, b_mean, g_mean,
                                                                                         label]
            else:
                print(feature_set, "IS NOT A VALID FEATURE SET.")

        return feature_dict


def extract_fixation_features(subject, f, feature_set, feature_dict, label_orig):
    """parse Matlab struct to extract EEG singals only for fixation occurring inside wordbounds"""
    """ extract features in the order they were read"""
    rawData = f['rawData']
    contentData = f['content']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in range(len(rawData)):
            # get word level data
            wordData = f['word']

            label = label_orig
            full_idx = len(feature_dict[feature_set])

            try:
                word_data = dlh.extract_word_level_data(f, f[wordData[idx][0]])

                fix_order_raw_eeg = []
                fixations_indices = {}
                # get EEG data in order of fixations
                for widx in word_data["word_reading_order"]:
                    if len(word_data[widx]["RAW_EEG"]) > 1:
                        if widx not in fixations_indices:
                            fixations_indices[widx] = 0
                        else:
                            fixations_indices[widx] += 1
                        fixation_avg = np.nanmean(word_data[widx]["RAW_EEG"][fixations_indices[widx]], axis=0)
                        #print(fixation_avg)
                    else:
                        fixation_avg = np.nanmean(word_data[widx]["RAW_EEG"][0], axis=0)
                    fix_order_raw_eeg.append(fixation_avg)

                #print(fix_order_raw_eeg[0])


                word_g1_electrodes = []; word_g2_electrodes = [];
                sent_feats = []; sent_trt_t1 = []; sent_trt_t2 = []; sent_trt_a1 = []; sent_trt_a2 = []; sent_trt_b1 = [];
                sent_trt_b2 = []; sent_trt_g1 = []; sent_trt_g2 = [];
                # get EEG data in order of words
                for widx in range(len(word_data)-1):
                    if word_data[widx]["RAW_EEG"]:
                        # "fix_avg_raw_eeg"
                        fixation_avg = [np.mean(fix) for fix in word_data[widx]["RAW_EEG"]]
                        sent_feats.append(np.nanmean(fixation_avg))
    
                        # "fix_electrode_features_gamma"
                        word_g1_electrodes.append(word_data[widx]["TRT_g1"])
                        word_g2_electrodes.append(word_data[widx]["TRT_g2"])

                if feature_set == "fix_electrode_features_gamma" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes, axis=0), np.nanmean(word_g2_electrodes, axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]
    
                elif feature_set == "fix_electrode_features_gamma_10%" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        #print(len(word_g1_electrodes), len(word_g2_electrodes))
                        # take 10%, but at least 1 word
                        p10 = max(round(len(word_g1_electrodes)/10), 1)
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes[:p10], axis=0), np.nanmean(word_g2_electrodes[:p10], axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]
    
                elif feature_set == "fix_electrode_features_gamma_20%" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        # take 10%, but at least 1 word
                        p20 = max(round(len(word_g1_electrodes)/5), 1)
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes[:p20], axis=0), np.nanmean(word_g2_electrodes[:p20], axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]

                elif feature_set == "fix_electrode_features_gamma_50%" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        # take 10%, but at least 1 word
                        p50 = max(round(len(word_g1_electrodes)/2), 1)
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes[:p50], axis=0), np.nanmean(word_g2_electrodes[:p50], axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]

                elif feature_set == "fix_electrode_features_gamma_75%" and word_g1_electrodes:
                    if not np.isnan(word_g1_electrodes).any():
                        # take 10%, but at least 1 word
                        p75 = max(round((len(word_g1_electrodes)/ 10)*7.5), 1)
                        feat_list = np.hstack((np.nanmean(word_g1_electrodes[:p75], axis=0), np.nanmean(word_g2_electrodes[:p75], axis=0)))
                        feature_dict[feature_set][subject + "_" + label_orig + "_" + str(idx)] = list(np.mean(feat_list, axis=1)) + [label_orig]

                elif feature_set == 'fix_order_raw_eeg_electrodes' and fix_order_raw_eeg:
                    avg = np.nanmean(fix_order_raw_eeg, axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_10%' and fix_order_raw_eeg:
                    p10 = max(round(len(fix_order_raw_eeg) / 10), 1) # at least 1 fixation if sentence contains <10
                    avg = np.nanmean(fix_order_raw_eeg[:p10], axis=0)
                    #print(avg)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_20%' and fix_order_raw_eeg:
                    p20 = max(round(len(fix_order_raw_eeg) / 5), 1)
                    avg = np.nanmean(fix_order_raw_eeg[:p20], axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_50%' and fix_order_raw_eeg:
                    p50 = max(round(len(fix_order_raw_eeg) / 2), 1)
                    avg = np.nanmean(fix_order_raw_eeg[:p50], axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = list(avg) + [label]

                elif feature_set == 'fix_order_raw_eeg_electrodes_75%' and fix_order_raw_eeg:
                    p75 = max(round((len(fix_order_raw_eeg) / 10)*7.5), 1)
                    avg = np.nanmean(fix_order_raw_eeg[:p75], axis=0)
                    if not np.isnan(avg).any():
                        feature_dict[feature_set][subject + "_" + label + "_" + str(idx) + "_" + str(full_idx)] = list(avg) + [label]

            except ValueError:
                print("NO WORD DATA AVAILABLE for sentence ", idx)

    return feature_dict
