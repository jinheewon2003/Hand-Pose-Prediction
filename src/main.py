from pathlib import Path
import numpy as np
from data_processing import process_data, read_pkl_file
from data_visualization import data_visualization, combine_videos
from ml_models import random_split, first_split, lstm_model
from sklearn.preprocessing import MinMaxScaler

# Files to process
files = ["p15-l2-s3-r2.pkl", "p15-l2-s3-r3.pkl"]
path_to_data = "data/"
path_to_adj_data = path_to_data + "adj_"
root_folder = Path(__file__).parents[1]

# Can block out after data is saved in data folder
#######
# Data Processing
for filename in files:
    filepath = root_folder / (path_to_data + filename)
    new_filepath = root_folder / (path_to_adj_data + filename)
    process_data(filepath, new_filepath)
#######

# Compile data across files
mphands_data = []
eit_data = []
for filename in files:
    filepath = root_folder / (path_to_adj_data + filename)
    data_to_add = read_pkl_file(filepath)
    mphands_data.extend(data_to_add['mphands_data'].tolist())
    eit_data.extend(data_to_add['eit_data'].tolist())
mphands_data = np.array(mphands_data)
eit_data = np.array(eit_data)

# Scale data to be more interpretable
scaler_eit_data = MinMaxScaler()
eit_data = scaler_eit_data.fit_transform(eit_data)

scaler_mphands_data = MinMaxScaler()
mphands_data = scaler_mphands_data.fit_transform(mphands_data)

# Can block out after visualization is saved in visualizations folder
#######
# Data Visualization of original data
filename = "data_visualization.mp4"
filepath = root_folder / ("visualizations/" + filename)
data_visualization(mphands_data, filepath, preview=False, speed = 750)
#######

# Split data to test - train; currently using random split
X_train, X_test, y_train, y_test = random_split(eit_data, mphands_data)

# Run data through LSTM model
y_pred = lstm_model(X_train, X_test, y_train, y_test)

# Data visualization of predicted model
filename = "random_split_lstm_model_predicted.mp4"
predicted_filepath = root_folder / ("visualizations/" + filename)
# data_visualization(y_pred, predicted_filepath, preview=False, speed = 750)
filename = "random_split_lstm_model_actual.mp4"
expected_filepath = root_folder / ("visualizations/" + filename)
# data_visualization(y_test, expected_filepath, preview=False, speed = 750)
filename = "random_split_lstm_model_compared.mp4"
compared_filepath = root_folder / ("visualizations/" + filename)
combine_videos(predicted_filepath, expected_filepath, compared_filepath)