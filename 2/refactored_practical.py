import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np

from scipy import sparse
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcess
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier

import matplotlib.pyplot as plt

TRAIN_DIR = "train"
TEST_DIR = "test"

call_set = set([])


malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for 
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))


def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)


def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def add_count(feature_dict, feature_name):
    if feature_name not in feature_dict:
        feature_dict[feature_name] = 1
    else:
        feature_dict[feature_name] += 1
    return feature_dict

def call_feats(tree):
    include_other_features = True
    include_first_calls = True
    include_second_calls = True
    include_call_pairs = False

    all_calls = ['recv_socket', 'create_open_file', 'sleep', 'open_scmanager', 'load_driver', 
              'get_host_by_addr', 'create_interface', 'create_mutex', 'set_value', 'enum_items', 
              'get_computer_name', 'read_value', 'write_value', 'change_service_config', 
              'copy_file', 'exit_windows', 'connect_share', 'enum_modules', 'bind_socket', 
              'enum_keys', 'delete_value', 'enum_types', 'open_service', 'processes', 
              'add_share', 'create_socket', 'enum_user', 'dump_line', 'unload_driver', 
              'enum_values', 'thread', 'load_dll', 'create_window', 'read_section_names', 
              'com_create_instance', 'message', 'get_userinfo', 'get_file_attributes', 'find_file', 
              'open_file', 'get_username', 'create_service', 'query_value', 'create_file', 
              'move_file', 'open_key', 'send_socket', 'vm_write', 'delete_file', 
              'create_process_as_user', 'get_system_time', 'create_mailslot', 'com_createole_object', 
              'listen_socket', 'enum_share', 'open_mutex', 'vm_protect', 'all_section', 
              'vm_mapviewofsection', 'get_windows_directory', 'enum_processes', 'open_url', 
              'download_file', 'com_get_class_object', 'kill_process', 'load_image', 'delete_share', 
              'create_process', 'logon_as_user', 'get_system_directory', 'set_thread_context', 
              'create_process_nt', 'destroy_window', 'vm_allocate', 'enum_handles', 'connect_socket', 
              'set_file_time', 'start_service', 'create_thread_remote', 'show_window', 'open_process', 
              'impersonate_user', 'connect', 'enum_services', 'process', 'vm_read', 'check_for_debugger', 
              'query_keyinfo', 'delete_service', 'read_section', 'enum_window', 'set_system_time', 
              'add_netjob', 'ping', 'set_windows_hook', 'control_service', 'accept_socket', 
              'trimmed_bytes', 'download_file_to_cache', 'find_window', 'get_host_by_name', 
              'set_file_attributes', 'revert_to_self', 'create_key', 'create_thread', 'enum_subtypes', 
              'delete_key', 'create_directory', 'remove_directory', 'create_namedpipe']

    first_calls = map(lambda call: "fc_" + call, all_calls)

    second_calls = map(lambda call: "sc_" + call, all_calls)

    call_pairs = map(lambda call_one: map(lambda call_two: call_one + "_" + call_two, second_calls), first_calls)

    all_features = all_calls

    if include_other_features:
        other_features = ['Administrator', 'SYSTEM', 'NETZWERKDIENST', 'LOKALER DIENST', 
                      'SCM', 'InjectedCode', 'SvcHost', 'CreateProcess', 'BHOInstalled', 'DCOMService', 'AnalysisTarget',
                     'NormalTermination', 'Unknown', 'KilledByWindowsLoader', 'Timeout']
        all_features += other_features
    if include_first_calls:
        all_features += first_calls
    if include_second_calls:
        all_features += second_calls
    if include_call_pairs:
        all_features += call_pairs


    if add_bigrams
        bigram_calls = []
        for a_call in all_calls:
            for b_call in all_calls:
                bigram_calls.append(a_call + '_' + b_call)
                
        combined_calls = all_calls + bigram_calls

    feature_dict = {}
    last_call='start'
    for el in tree.iter():
        call = el.tag
        feature_dict = add_count(feature_dict, call)
            
    root = tree.getroot()
    for process in root.findall('process[@username]'):
        username = process.attrib['username']
        feature_dict = add_count(feature_dict, username)
    for process in root.findall('process[@startreason]'):
        startreason = process.attrib['startreason']
        feature_dict = add_count(feature_dict, startreason)
    for process in root.findall('process[@terminationreason]'):
        terminationreason = process.attrib['terminationreason']
        feature_dict = add_count(feature_dict, terminationreason)
    j = 0

    for call in root.findall('process/thread/all_section/'):
        name = call.tag
        if (name != 'load_image') and (name != 'load_dll'): 
            j += 1
            if j == 1:
                first_call = 'fc_' + name
                feature_dict = add_count(feature_dict, first_call)
            if j == 2:
                second_call = 'sc_' + name
                feature_dict = add_count(feature_dict, second_call)
                break

    # Attempt to add call-pairs, which, in theory, help account for sequencing of calls.
    # This perhaps should be extended to call-triplets, etc.
    for idx, call in enumerate(root.findall('process/thread/all_section/')):
        if idx == 1:
            second_call = call.tag
        elif idx >= 2:
            first_call = second_call
            second_call = call.tag
            call_pair = "fc_" + first_call + "_sc_" + second_call
            feature_dict = add_count(feature_dict, call_pair)
                
        if include_call_pairs:                
            # Bigram Calls
            combo = last_call + '_' + call
            if combo not in call_counter:
                call_counter[combo] = 1
            else:
               call_counter[combo] += 1
            last_call = call
            
    call_feat_array = np.zeros(len(all_features))
    for i in range(len(all_features)):
        call = all_features[i]
        call_feat_array[i] = 0
        if call in feature_dict:
            call_feat_array[i] = feature_dict[call]

    return call_feat_array

def write_to_file(filename, ids, predictions):
    zips = zip(ids, predictions)
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(zips):
            f.write(str(p[0]) + "," + str(p[1]) + "\n")

def main():
    X_train_all, t_train_all, train_all_ids = create_data_matrix(0, 3086, TRAIN_DIR)
    X_train, X_valid, t_train, t_valid = train_test_split(X_train_all, t_train_all, test_size=0.20, random_state=37)
    X_test_all, t_test_all, test_all_ids = create_data_matrix(0, 3724, TEST_DIR)

    sv = svm.SVC(kernel='poly')
    sv.fit(X_train, t_train)
    print "SVM Score was: %f" % clf.score(X_valid, t_valid)

    rf = RandomForestClassifier(n_estimators=30, min_samples_split=1, random_state=37)
    rf.fit(X_train, t_train)
    print "RandomForest Score was: %f" % (rf.score(X_valid, t_valid))

    lr = LogisticRegression(penalty='l2',solver='newton-cg',max_iter=500)
    lr.fit(X_train, t_train)
    print "LogisticRegression Score was: %f" % (lr.score(X_valid, t_valid))

    clf = GaussianNB()
    clf.fit(X_train, t_train)
    print "GaussianNB Score was: %f" % (clf.score(X_valid, t_valid))

    nn = KNeighborsClassifier(n_neighbors=6, weights='uniform')
    nn.fit(X_train, t_train)
    score = nn.score(X_valid, t_valid)
    print "KNeighbors Score was: %f" % (score)

    rnc = RadiusNeighborsClassifier(radius=6,outlier_label=8, p=2)
    rnc.fit(X_train, t_train)
    print "RadiusNeighbors Score was: %f" % (rnc.score(X_valid, t_valid))

    # Get predictions
    rf = RandomForestClassifier(n_estimators=30, min_samples_split=1)
    rf.fit(X_train_all, t_train_all)
    test_predictions = rf.predict(X_test_all)

    write_to_file("prediction.csv", test_all_ids, test_predictions)
    
    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()
    