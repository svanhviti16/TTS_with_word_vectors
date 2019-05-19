import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
import os, sys
import numpy as np
import sklearn

# ----------------------------------------------
# FINAL PROJECT IN SPEECH SYNTHESIS, SPRING 2019
# Svanhvít Lilja Ingólfsdóttir
# Reykjavik University
# ----------------------------------------------

# binary files (.lab_dur)
bin_dir = "../Ossian_project/hlbsf_101_800_test/bin_lab_phone_no_sil_347/"
# xml files (.utt)
utt_dir = "../Ossian_project/hlbsf_101_800_test/utt/"
# word embeddings file
embeddings_file = "../Ossian_project/WV/embeddings.txt"

dimension_size = 347 # taken from the file name of binary files

# ---------
# FUNCTIONS
# ---------

# Merlin function from io_funcs in Ossian/tools/src/merlin/io_funcs/binary_io. 
# Reads a binary file into a numpy array, given the file and the dimension
def load_binary_file(file_name, dimension):
        fid_lab = open(bin_dir+file_name+".lab_dur", 'rb')
        #extracts array from bin file
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        #print(f" features size {features.size} + {features}")
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return  features

def concatenate_arrays(bin_array, wv_array, file_name):
    #print(bin_array)
    #print(wv_array)
    #print(f'bin {np.size(bin_array, 1) }, wv {np.size(wv_array, 1)}')
    assert np.size(bin_array, 0) == np.size(wv_array, 0)
    concatenated = np.concatenate((bin_array, wv_array), axis=1)
    concatenated.tofile("concatenated_bin/"+file_name+".lab_dur")

# Makes a lookup in the embeddings.txt file by word and returns the corresponding list of matrix values
def find_word(key):
    if key in wv_dict:
        return(wv_dict.get(key))

# returns the mean of the wv values, for OOV words
def get_mean_array():
    np_arr = np.array(list(wv_dict.values()))
    return np_arr.mean(axis=0)

def get_frame_count_for_utt(tokens):
    total_frames = 0
    for token in tokens:
        # skip empty tokens (with no start and end times)
        if token.attrib.get("start"):
            if token.attrib.get("has_silence") == "no":
                duration = int(token.attrib.get("end")) - int(token.attrib.get("start"))
                frame_num = int(duration/5)
                total_frames += frame_num
    #print(total_frames)
    return total_frames

def get_segment_count_for_token(token):
    segments_in_token = 0
    for s in token.iter('segment'):
        segments_in_token += 1
    #print("segment count in token " + str(segments_in_token))
    return segments_in_token

# --------------------
# LOADING WORD VECTORS
# Reads a word embeddings file into a dictionary of words and their embeddings 
# --------------------
wv_dict = {}
with open(embeddings_file) as fi:
    corpus = fi.readlines()
    lines = [line.rstrip('\n') for line in corpus]
for line in lines:
    number_arr = []
    #print(line)
    number_arr.append([float(x) for x in line.split()[1:]])
        #arr.append(number)
    # word is the key, real number matrix is the value
    wv_dict.update({line.split()[0] : number_arr[0]})

# ---------------
# XML PARSING
# utt xml folder
# ---------------
# each utterance in utt
for f in os.listdir(utt_dir):
    tree = ET.parse(utt_dir + f)
    root = tree.getroot()
    utt_filename = f.split(".")[0]

    # passing corresponding utt filename for each bin file
    bin_array = load_binary_file(utt_filename, dimension_size)

    # parsing the xml using etree
    token_elements = root.findall(".//token")
    
    total_segments = 0
    for token in token_elements:
        total_segments += get_segment_count_for_token(token)
    #total_frame_count = get_frame_count_for_utt(token_elements)

    # create an array full of zeros (default)
    zero_arr = np.zeros((total_segments, 200), dtype=float)

    # repeat for each word
    # want to write the vector for the word once for each frame of the token
    row = 0
    for token in token_elements:    
        # skip empty tokens (with no start and end times)
        if token.attrib.get("start"):
            if token.attrib.get("has_silence") == "no":
                # ended up not using the frame count anywhere, which may explain the error:
                #duration = int(token.attrib.get("end")) - int(token.attrib.get("start"))
                #frame_num = int(duration/5)

                for i in range(0, get_segment_count_for_token(token)):

                    # need to check whether the word appears in the embeddings file
                    if (find_word(token.attrib.get("text"))):
                        np_arr = np.array(find_word(token.attrib.get("text")))

                    # for OOV words, we use the mean array
                    else:
                        np_arr = get_mean_array()

                    #  append row to zero array ( check file 1261 )
                    try:
                        zero_arr[row, :] = np_arr
                    except IndexError:
                        pass

                    row += 1
                    #print("row " + str(row))
            # for silence tokens
            else:
                row += 1
    
    concatenate_arrays(bin_array, zero_arr, utt_filename)
    print("Utterance " + utt_filename + " processed")
    

