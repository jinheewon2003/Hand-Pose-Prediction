import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from data_processing import read_pkl_file
import re

def grouped_files(folder_path):
    info = {}
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            delimiters = "-eit", "-mphands", "(c)"
            regex_pattern = '|'.join(map(re.escape, delimiters))
            group = re.split(regex_pattern, filename)[0]
            if group in info:
                info[group].append(filename)
            else:
                info[group] = [filename]
    return info

# Files to process
root_folder = Path(__file__).parents[1]
data_directory = root_folder / "separated_data/"
filegroups = grouped_files(data_directory)

for file_root in filegroups:
    files = filegroups[file_root]
    
    if "eit" in files[0]:
        store = files[0]
        files[0] = files[1]
        files[1] = store
    
    mphands_file_path = root_folder / ("separated_data/" + files[0])
    mphands_df = read_pkl_file(mphands_file_path)
    mphands_data_list = mphands_df['data'].tolist()
    mphands_data_start_time_list = mphands_df['timestamp_start'].tolist()
    mphands_data_end_time_list = mphands_df['timestamp_end'].tolist()
    
    eit_file_path = root_folder / ("separated_data/" + files[1])
    eit_df = read_pkl_file(eit_file_path)
    eit_data_list = eit_df['data'].tolist()
    eit_data_time_list = eit_df['timestamp'].tolist()
    
    
    # # Creates CSV files of data for readability 
    # with open(mphands_file_path, "rb") as f:
    #     object = pickle.load(f)
    
    # save_to = root_folder / ("csv_data/" + files[0] + ".csv")
    # df = pd.DataFrame(object)
    # df.to_csv(save_to)
    
    # with open(eit_file_path, "rb") as f:
    #     object = pickle.load(f)
    
    # save_to = root_folder / ("csv_data/" + files[1] + ".csv")
    # df = pd.DataFrame(object)
    # df.to_csv(save_to)
    
    mphands_data = []
    eit_data = []
    row_num = 0
    
    # for index, row in enumerate(eit_data_list):
    #     while (len(mphands_data_list) > row_num + 1 and mphands_data_end_time_list[row_num] < eit_data_time_list[index]):
    #         row_num += 1
        
    #     if not np.isnan(np.array(mphands_data_list[row_num][0])).any():
    #         eit_data.append(eit_data_list[index])
    #         mphands_data.append(np.array(mphands_data_list[row_num][0]).flatten().tolist())
    #     elif not np.isnan(np.array(mphands_data_list[row_num][1])).any():
    #         eit_data.append(eit_data_list[index])
    #         mphands_data.append(np.array(mphands_data_list[row_num][1]).flatten().tolist())
            
    for index, row in enumerate(mphands_data_list):
        # Calculate eit value for array
        count = 0
        sums = np.zeros_like(eit_data_list[0])
        while (len(eit_data_time_list) > row_num and eit_data_time_list[row_num] < mphands_data_end_time_list[index]):
            sums += eit_data_list[row_num]
            row_num += 1
            count += 1
        
        if count == 0:
            continue
        if np.isnan(np.array(row[0])).any() and np.isnan(np.array(row[1])).any():
            continue
        
        eit_data.append((sums / count).tolist())
        if not np.isnan(np.array(row[0])).any():
            mphands_data.append(np.array(row[0]).flatten().tolist())
        else: 
            mphands_data.append(np.array(row[1]).flatten().tolist())
    
    new_df = pd.DataFrame({'mphands_data': mphands_data, 'eit_data': eit_data})
    new_df['mphands_data'] = mphands_data
    new_df['eit_data'] = eit_data

    # Creates CSV files of data for readability 
    save_to = root_folder / ("csv_data/" + file_root + ".csv")
    df = pd.DataFrame(new_df)
    df.to_csv(save_to)

    new_file_name = root_folder / ("data/" + file_root + ".pkl")
    with open(new_file_name, 'wb') as f:
        pickle.dump(new_df, f)