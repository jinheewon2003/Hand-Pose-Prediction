# Map the rows based on the mapping
import pickle
import numpy as np

# Load the original pkl file
pkl_file_path = 'p15-l2-s3-r2.pkl'

def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

df = read_pkl_file(pkl_file_path)

# Change to list
mphands_data_list = np.array(df['mphands_data'].tolist())

# Define the mapping (new array pos: old array pos)
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

new_pkl_file = 'adj_' + pkl_file_path
with open(new_pkl_file, 'wb') as f:
    pickle.dump(new_df, f)


# Also try non-randomized time shuffle in ML model (ie change how test + train are determined)
