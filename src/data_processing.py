import pickle
import numpy as np
from pathlib import Path


def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def process_data(file_name, new_file_name):
    # Open file and get data
    df = read_pkl_file(file_name)

    # Change to list
    mphands_data_list = np.array(df['mphands_data'].tolist())

    # Define the mapping (new array pos: old array pos) based on 21 keypoints and original data
    mapping = {0: 20, 1: 16, 2: 17, 3: 19, 4: 18, 5: 0, 6: 1, 7: 3, 8: 2, 9: 4, 10: 5, 11: 7, 12: 6, 13: 12, 14: 13, 15: 15, 16: 14, 17: 8, 18: 9, 19: 11, 20: 10}

    # Apply the mapping to rearrange the values within each row
    mapped_mphands_data_list = np.array([
        [[row[mapping.get(i)*3]] + [row[mapping.get(i)*3+1]] + [row[mapping.get(i)*3+2]] for i in range(int(mphands_data_list.shape[1]/3))]
        for row in mphands_data_list
    ])

    # Flatten
    flattened_list = mapped_mphands_data_list.flatten()

    # Rearrange
    reshaped_array = flattened_list.reshape(mphands_data_list.shape)

    # Save the modified PKL file
    new_df = df.copy()  # Create a copy of the original DataFrame
    new_df['mphands_data'] = reshaped_array.tolist()

    print(new_file_name)
    with open(new_file_name, 'wb') as f:
        pickle.dump(new_df, f)