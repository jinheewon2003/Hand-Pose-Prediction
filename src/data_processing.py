import pickle
import numpy as np
from pathlib import Path
import json
import os

def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_list_to_json_file(lst, filename):
    with open(filename, 'w') as file:
        json.dump(lst, file)

def grouped_files(folder_path):
    info = {}
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            group = filename.split("-r")[0]
            if group in info:
                info[group].append(filename)
            else:
                info[group] = [filename]
    return info

def process_data(file_name, new_file_name, mapping):
    # Open file and get data
    df = read_pkl_file(file_name)

    # Change to list
    mphands_data_list = np.array(df['mphands_data'].tolist())

    # Apply the mapping to rearrange the values within each row
    mapped_mphands_data_list = np.array([
        [[row[mapping.get(i)*3]] + [row[mapping.get(i)*3+1]] + [row[mapping.get(i)*3+2]] for i in range(int(mphands_data_list.shape[1]/3))]
        for row in mphands_data_list
    ])

    # Flatten
    flattened_list = mapped_mphands_data_list.flatten()

    # Rearrange
    reshaped_array = flattened_list.reshape(mphands_data_list.shape)
    
    palm_size = np.linalg.norm(reshaped_array[0][0] - reshaped_array[0][9])
    # Scale model to hand size
    av_hand_size_cm = 8.5
    reshaped_array = reshaped_array * av_hand_size_cm / palm_size

    # Save the modified PKL file
    new_df = df.copy()  # Create a copy of the original DataFrame
    new_df['mphands_data'] = reshaped_array.tolist()

    print(new_file_name)
    with open(new_file_name, 'wb') as f:
        pickle.dump(new_df, f)