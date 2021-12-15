import h5py
import numpy as np
import pickle as pkl
import data_loading_helpers as dlh
import re
import config

eeg_float_resolution = np.float16

Alpha_ffd_names = ['FFD_a1', 'FFD_a1_diff', 'FFD_a2', 'FFD_a2_diff']
Beta_ffd_names = ['FFD_b1', 'FFD_b1_diff', 'FFD_b2', 'FFD_b2_diff']
Gamma_ffd_names = ['FFD_g1', 'FFD_g1_diff', 'FFD_g2', 'FFD_g2_diff']
Theta_ffd_names = ['FFD_t1', 'FFD_t1_diff', 'FFD_t2', 'FFD_t2_diff']
Alpha_gd_names = ['GD_a1', 'GD_a1_diff', 'GD_a2', 'GD_a2_diff']
Beta_gd_names = ['GD_b1', 'GD_b1_diff', 'GD_b2', 'GD_b2_diff']
Gamma_gd_names = ['GD_g1', 'GD_g1_diff', 'GD_g2', 'GD_g2_diff']
Theta_gd_names = ['GD_t1', 'GD_t1_diff', 'GD_t2', 'GD_t2_diff']
Alpha_gpt_names = ['GPT_a1', 'GPT_a1_diff', 'GPT_a2', 'GPT_a2_diff']
Beta_gpt_names = ['GPT_b1', 'GPT_b1_diff', 'GPT_b2', 'GPT_b2_diff']
Gamma_gpt_names = ['GPT_g1', 'GPT_g1_diff', 'GPT_g2', 'GPT_g2_diff']
Theta_gpt_names = ['GPT_t1', 'GPT_t1_diff', 'GPT_t2', 'GPT_t2_diff']
Alpha_sfd_names = ['SFD_a1', 'SFD_a1_diff', 'SFD_a2', 'SFD_a2_diff']
Beta_sfd_names = ['SFD_b1', 'SFD_b1_diff', 'SFD_b2', 'SFD_b2_diff']
Gamma_sfd_names = ['SFD_g1', 'SFD_g1_diff', 'SFD_g2', 'SFD_g2_diff']
Theta_sfd_names = ['SFD_t1', 'SFD_t1_diff', 'SFD_t2', 'SFD_t2_diff']
Alpha_trt_names = ['TRT_a1', 'TRT_a1_diff', 'TRT_a2', 'TRT_a2_diff']
Beta_trt_names = ['TRT_b1', 'TRT_b1_diff', 'TRT_b2', 'TRT_b2_diff']
Gamma_trt_names = ['TRT_g1', 'TRT_g1_diff', 'TRT_g2', 'TRT_g2_diff']
Theta_trt_names = ['TRT_t1', 'TRT_t1_diff', 'TRT_t2', 'TRT_t2_diff']
# IF YOU CHANGE THOSE YOU MUST ALSO CHANGE CONSTANTS
Alpha_features = Alpha_ffd_names + Alpha_gd_names + Alpha_gpt_names + Alpha_trt_names# + Alpha_sfd_names
Beta_features = Beta_ffd_names + Beta_gd_names + Beta_gpt_names + Beta_trt_names# + Beta_sfd_names
Gamma_features = Gamma_ffd_names + Gamma_gd_names + Gamma_gpt_names + Gamma_trt_names# + Gamma_sfd_names
Theta_features = Theta_ffd_names + Theta_gd_names + Theta_gpt_names + Theta_trt_names# + Theta_sfd_names


def do_print(string, file):
    """
    Prints on scree and on given file simultaneously

    :param string:  (str)   String to print
    :param file:    (file)  File on which to save
    :return:
        None
    """
    print(string)
    print(string, file = file)


def is_real_word(word):
    """
    Check if the word is a real word
    :param word:    (str)   word string
    :return:
        is_word (bool)  True if it is a real word
    """
    is_word = re.search('[a-zA-Z0-9]', word)
    return is_word


def open_subject_sentence_data(subject):
    """
    Opens data and returns h5py object
    :param subject:     (str)   Subject of interest code name (e.g. ZPH)
    :return:
        f       (h5py)  Subject's data
    """
    filepath = "data_to_preprocess/results" + subject + "_SR.mat"
    f = h5py.File(filepath)
    return f


def load_matlab_string(matlab_extracted_object):
    """
    Converts a string loaded from h5py into a python string
    :param matlab_extracted_object:     (h5py)  matlab string object
    :return:
        extracted_string    (str)   translated string
    """
    extracted_string = u''.join(chr(c) for c in matlab_extracted_object)
    return extracted_string

"""
bad_channels_data = pd.read_csv("../eeg-sentiment/eeg-quality/badChannelsEEG.csv")

def get_bad_channels(idx, subject, task = "sentiment"):
    # TODO: Discuss and fix bad_channels loading
    session = int(idx/50) + 1
    if task == "sentiment":
        session_file_name1 = "SR" + str(session)
        session_file_name2 = "SNR" + str(session)
    else:
        raise Exception("only sentiment task available so far")
    file_filter = (bad_channels_data['file'] == session_file_name1) | (bad_channels_data['file'] == session_file_name2)
    subject_filter = bad_channels_data['sj'] == subject
    bad_channels = bad_channels_data.loc[file_filter & subject_filter]["bad_channels"]
    bad_channels = bad_channels.values[0].split(" ") if bad_channels.values else None
    return bad_channels
"""

def extract_word_order_from_fixations(fixations_order_per_word):
    """
    Extracts fixation order for a specific sentence
    Example:
            input:  [[0, 3], [], [1, 4], [2]]
            output: [0, 2, 3, 0, 2]

    :param fixations_order_per_word:    (list)  Contains one list for each word in the sentence, each list
                                                representing the fixation numbers on word w
    :return:
        words_fixated_in_order:     (list)  Contains integers representing the word fixated at fixation f
    """
    if not fixations_order_per_word:
        return []
    fxs_list = [list(fixs) if len(fixs.shape)>1 else [] for fixs in fixations_order_per_word]
    n_tot_fixations = len(sum(fxs_list, []))
    words_fixated_in_order = []
    for fixation_n in range(n_tot_fixations):
        mins_per_word_idx = np.array([min(i) if len(i)>0 else np.nan for i in fxs_list])
        next_word_fixated = int(np.nanargmin(mins_per_word_idx)) # Seems to work like this
        fxs_list[next_word_fixated].remove(min(fxs_list[next_word_fixated]))
        words_fixated_in_order.append(next_word_fixated)
    return words_fixated_in_order


def extract_word_level_data(data_container, word_objects, eeg_float_resolution = np.float16):
    """
    Extracts word level data for a specific sentence

    :param data_container:          (h5py)  Container of the whole data, h5py object
    :param word_objects:            (h5py)  Container of all word data for a specific sentence
    :param eeg_float_resolution:    (type)  Resolution with which to save EEG, used for data compression
    :return:
        word_level_data     (dict)  Contains all word level data indexed by their index number in the sentence,
                                    together with the reading order, indexed by "word_reading_order"
    """
    available_objects = list(word_objects)
    #print(available_objects)
    contentData = word_objects['content']
    fixations_order_per_word = []
    if "rawEEG" in available_objects:

        rawData = word_objects['rawEEG']
        #icaData = word_objects['IC_act_automagic']
        etData = word_objects['rawET']

        ffdData = word_objects['FFD']
        gdData = word_objects['GD']
        gptData = word_objects['GPT']
        trtData = word_objects['TRT']
        nFixData = word_objects['nFixations']
        fixPositions = word_objects["fixPositions"]
        trt_t1Data = word_objects['TRT_t1']
        trt_t2Data = word_objects['TRT_t2']
        trt_a1Data = word_objects['TRT_a1']
        trt_a2Data = word_objects['TRT_a2']
        trt_b1Data = word_objects['TRT_b1']
        trt_b2Data = word_objects['TRT_b2']
        trt_g1Data = word_objects['TRT_g1']
        trt_g2Data = word_objects['TRT_g2']

        #Alpha_features_data = [word_objects[feature] for feature in Alpha_features]
        #Beta_features_data = [word_objects[feature] for feature in Beta_features]
        #Gamma_features_data = [word_objects[feature] for feature in Gamma_features]
        #Theta_features_data = [word_objects[feature] for feature in Theta_features]

        assert len(contentData) == len(etData) == len(rawData), "different amounts of different data!!"

        zipped_data = zip(rawData, etData, contentData, ffdData, gdData, gptData, trtData, nFixData, fixPositions, trt_t1Data, trt_t2Data, trt_a1Data, trt_a2Data, trt_b1Data, trt_b2Data, trt_g1Data, trt_g2Data)
        word_level_data = {}
        word_idx = 0
        for raw_eegs_obj, ets_obj, word_obj, ffd, gd, gpt, trt, nFix, fixPos, trt_t1, trt_t2, trt_a1, trt_a2, trt_b1, trt_b2, trt_g1, trt_g2 in zipped_data:
            word_string = load_matlab_string(data_container[word_obj[0]])
            #if is_real_word(word_string):
            data_dict = {}
            data_dict["RAW_EEG"] = extract_all_fixations(data_container, raw_eegs_obj[0], eeg_float_resolution)
            #data_dict["ICA_EEG"] = extract_all_fixations(data_container, ica_eegs_obj[0], eeg_float_resolution)
            data_dict["RAW_ET"] = extract_all_fixations(data_container, ets_obj[0], np.float32)
            data_dict["FFD"] = data_container[ffd[0]].value[0, 0] if len(data_container[ffd[0]].value.shape) == 2 else None
            data_dict["GD"] = data_container[gd[0]].value[0, 0] if len(data_container[gd[0]].value.shape) == 2 else None
            data_dict["GPT"] = data_container[gpt[0]].value[0, 0] if len(data_container[gpt[0]].value.shape) == 2 else None
            data_dict["TRT"] = data_container[trt[0]].value[0, 0] if len(data_container[trt[0]].value.shape) == 2 else None
            data_dict["nFix"] = data_container[nFix[0]].value[0, 0] if len(data_container[nFix[0]].value.shape) == 2 else None
            #print(np.array(data_container[trt_t1[0]]))
            #print(data_container[trt_t1[0]].value.shape)
            data_dict["TRT_t1"] = data_container[trt_t1[0]].value if len(data_container[trt_t1[0]].value.shape) == 2 else None
            data_dict["TRT_t2"] = data_container[trt_t2[0]].value if len(data_container[trt_t2[0]].value.shape) == 2 else None
            data_dict["TRT_a1"] = data_container[trt_a1[0]].value if len(
                data_container[trt_a1[0]].value.shape) == 2 else None
            data_dict["TRT_a2"] = data_container[trt_a2[0]].value if len(
                data_container[trt_t2[0]].value.shape) == 2 else None
            data_dict["TRT_b1"] = data_container[trt_b1[0]].value if len(
                data_container[trt_b1[0]].value.shape) == 2 else None
            data_dict["TRT_b2"] = data_container[trt_b2[0]].value if len(
                data_container[trt_b2[0]].value.shape) == 2 else None
            data_dict["TRT_g1"] = data_container[trt_g1[0]].value if len(
                data_container[trt_g1[0]].value.shape) == 2 else None
            data_dict["TRT_g2"] = data_container[trt_g2[0]].value if len(
                data_container[trt_g2[0]].value.shape) == 2 else None

            fixations_order_per_word.append(np.array(data_container[fixPos[0]]))

            #print([data_container[obj[word_idx][0]].value for obj in Alpha_features_data])

            """
            data_dict["ALPHA_EEG"] = np.concatenate([data_container[obj[word_idx][0]].value
                                                     if len(data_container[obj[word_idx][0]].value.shape) == 2 else []
                                                     for obj in Alpha_features_data], 0)

            data_dict["BETA_EEG"] = np.concatenate([data_container[obj[word_idx][0]].value
                                                    if len(data_container[obj[word_idx][0]].value.shape) == 2 else []
                                                    for obj in Beta_features_data], 0)

            data_dict["GAMMA_EEG"] = np.concatenate([data_container[obj[word_idx][0]].value
                                                     if len(data_container[obj[word_idx][0]].value.shape) == 2 else []
                                                     for obj in Gamma_features_data], 0)

            data_dict["THETA_EEG"] = np.concatenate([data_container[obj[word_idx][0]].value
                                                     if len(data_container[obj[word_idx][0]].value.shape) == 2 else []
                                                     for obj in Theta_features_data], 0)
            """


            data_dict["word_idx"] = word_idx
            # TODO: data_dict["word2vec_idx"] = Looked up after through the actual word.
            data_dict["content"] = word_string
            word_level_data[word_idx] = data_dict
            word_idx += 1
            #else:
             #   print(word_string + " is not a real word.")
    else:
        # If there are no word-level data it will be word embeddings alone
        word_level_data = {}
        word_idx = 0
        for word_obj in contentData:
            word_string = load_matlab_string(data_container[word_obj[0]])
            #if is_real_word(word_string):
            data_dict = {}
            #TODO: Make sure it was a good call to convert the below from {} to None
            data_dict["RAW_EEG"] = []
            data_dict["ICA_EEG"] = []
            data_dict["RAW_ET"] = []
            data_dict["FFD"] = None
            data_dict["GD"] = None
            data_dict["GPT"] = None
            data_dict["TRT"] = None
            data_dict["nFix"] = None
            #data_dict["ALPHA_EEG"] = []
            #data_dict["BETA_EEG"] = []
            #data_dict["GAMMA_EEG"] = []
            #data_dict["THETA_EEG"] = []

            data_dict["word_idx"] = word_idx
            data_dict["content"] = word_string
            word_level_data[word_idx] = data_dict
            word_idx += 1
            #else:
             #   print(word_string + " is not a real word.")
        sentence = " ".join([load_matlab_string(data_container[word_obj[0]]) for word_obj in word_objects['content']])
        #print("Only available objects for the sentence '{}' are {}.".format(sentence, available_objects))
    word_level_data["word_reading_order"] = extract_word_order_from_fixations(fixations_order_per_word)
    return word_level_data


def extract_all_fixations(data_container, word_data_object, float_resolution = np.float16):
    """
    Extracts all fixations from a word data object

    :param data_container:      (h5py)  Container of the whole data, h5py object
    :param word_data_object:    (h5py)  Container of fixation objects, h5py object
    :param float_resolution:    (type)  Resolution to which data re to be converted, used for data compression
    :return:
        fixations_data  (list)  Data arrays representing each fixation

    """
    word_data = data_container[word_data_object]
    fixations_data = []
    if len(word_data.shape) > 1:
        for fixation_idx in range(word_data.shape[0]):
            fixations_data.append(np.array(data_container[word_data[fixation_idx][0]]).astype(float_resolution))
    return fixations_data

def extract_sentence_level_data(subject, eeg_float_resolution=np.float16):
    """
    Load data dictionary from h5py object for one specific subject

    :param subject:                 (str)   Subject's code name
    :param eeg_float_resolution:    (type)  Resolution with which to save EEG, used for data compression
    :return:
        sentence_level_data (dict)  Dictionary containing all data for a specific subject
    """
    # TODO: consider adding smoothing for signals here (possibly full preprocessing too, e.g. normalization)
    f = open_subject_sentence_data(subject)
    sentence_data = f['sentenceData']
    rawData = sentence_data['rawData']
    icaData = sentence_data['IC_act_automagic']
    contentData = sentence_data['content']
    wordData = sentence_data['word']
    dataset, x, x_text, y, _ = dlh.get_processed_dataset(dataset_path="data/sentences", binary=False, verbose=True,
                                                         labels_from=None)
    sentence_order = dlh.get_sentence_order(dataset)
    sentence_level_data = {}
    for idx in range(len(rawData)): # raw data is an used but they all should be the same in length (400 for ternary, about 2/3 of that for binary)
        data_dict = {}
        obj_reference_raw = rawData[idx][0]
        data_dict["RAW_EEG"] = np.array(f[obj_reference_raw]).astype(eeg_float_resolution)
        obj_reference_ica = icaData[idx][0]
        data_dict["ICA_EEG"] = np.array(f[obj_reference_ica]).astype(eeg_float_resolution)
        obj_reference_content = contentData[idx][0]
        data_dict["content"] = load_matlab_string(f[obj_reference_content])
        # do_print(data_dict["content"], report_file)
        data_dict["sentence_number"] = idx
        label_idx = np.where(np.array(sentence_order) == idx)[0][0]
        data_dict["label"] = np.array(y[label_idx])
        data_dict["word_embedding_idxs"] = np.array(x[label_idx, :])
        data_dict["label_content"] = dataset['data'][label_idx]
        label_n = np.where(data_dict["label"] == 1)[0][0]
        data_dict["label_name"] = dataset['target_names'][label_n]
        bad_channels = get_bad_channels(idx, subject)
        data_dict["bad_channels"] = bad_channels.split(" ") if type(bad_channels) == str else None
        data_dict["word_level_data"] = extract_word_level_data(f, f[wordData[idx][0]], eeg_float_resolution=eeg_float_resolution)
        sentence_level_data[idx] = data_dict
    return sentence_level_data

def create_all_subjects_data(filename, eeg_float_resolution=np.float16):
    """
    Creates all subject training data dictionaries and saves them via pickle named filename_subject.pickle
    :param filename:                (str)   Name to use when saving the files
    :param eeg_float_resolution:    (type)  Format with which to save EEG, used for compression purposes
    :return:
        all_subjects_dict   (dict)  Dictionary containing all training data dictionaries (currently unused)
    """
    all_subjects_dict = {}
    for subject in config.subjects:
        print(subject)
        all_sentences_info = extract_sentence_level_data(subject, eeg_float_resolution=eeg_float_resolution)
        all_subjects_dict[subject] = all_sentences_info
        subject_file = filename + "_" + subject + ".pickle"
        print("Data saved in file " + subject_file)
        with open(subject_file, "wb") as f:
            pkl.dump(all_sentences_info, f)
    return all_subjects_dict