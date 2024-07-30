from pathlib import Path
import numpy as np
from data_processing import process_data, read_pkl_file, save_list_to_json_file, grouped_files
from data_visualization import data_visualization, combine_videos
from ml_models import random_split, first_split, lstm_model
from sklearn.preprocessing import MinMaxScaler

def workflow(files, file_root, root_folder, path_to_data = "data/", path_to_adj_data = "adj_data/"):
    # Can block out after data is saved in data folder
    #######
    mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19: 19, 20:20}
    # mapping = {0: 20, 1: 16, 2: 17, 3: 19, 4: 18, 5: 0, 6: 1, 7: 3, 8: 2, 9: 4, 10: 5, 11: 7, 12: 6, 13: 12, 14: 13, 15: 15, 16: 14, 17: 8, 18: 9, 19: 11, 20: 10}
    # Data Processing
    for filename in files:
        filepath = root_folder / (path_to_data + filename)
        new_filepath = root_folder / (path_to_adj_data + filename)
        
        # Scales data to be hand size as well
        process_data(filepath, new_filepath, mapping)
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

    # Data Visualization of original data
    #######
    # filename = file_root + ".mp4"
    # filepath = root_folder / ("visualizations/" + filename)
    # data_visualization(mphands_data, file_to_save=filepath, preview=False, speed = 500) # file_to_save=filepath
    #######
    
    
    # Scale data to be more interpretable
    scaler_eit_data = MinMaxScaler()
    eit_data = scaler_eit_data.fit_transform(eit_data)

    scaler_mphands_data = MinMaxScaler()
    mphands_data = scaler_mphands_data.fit_transform(mphands_data)
    
    # Split data to test - train; currently using random split
    X_train, X_test, y_train, y_test = first_split(eit_data, mphands_data)

    # Run data through LSTM model
    y_pred, overall_loss, point_errors = lstm_model(X_train, X_test, y_train, y_test)

    # Unscale model
    y_pred = scaler_mphands_data.inverse_transform(y_pred)
    y_test = scaler_mphands_data.inverse_transform(y_test)

    # Store predidcted mphands data
    filename = file_root + "_predicted.l"
    filepath = root_folder / ("predicted_ys/" + filename)
    y_pred_saved = y_pred.reshape(-1, y_test.shape[1])
    # print(y_pred.size)
    # print(y_test.size)
    # print(y_train.size)
    save_list_to_json_file(y_pred_saved.tolist(), filepath)

    # Store errors
    filename = file_root + "_predicted.l"
    filepath = root_folder / ("errors/" + filename)
    combined_errors = [overall_loss, [point_errors.tolist()]]
    save_list_to_json_file(combined_errors, filepath)

    # Data visualization of predicted model
    # filename = file_root + "_first_split_lstm_model_predicted.mp4"
    # predicted_filepath = root_folder / ("visualizations/" + filename)
    # data_visualization(y_pred, predicted_filepath, preview=False, speed = 750)

    # filename = file_root + "_first_split_lstm_model_actual.mp4"
    # expected_filepath = root_folder / ("visualizations/" + filename)
    # data_visualization(y_test, expected_filepath, preview=False, speed = 750)

    # filename = file_root + "_first_split_lstm_model_compared.mp4"
    # compared_filepath = root_folder / ("visualizations/" + filename)
    # combine_videos(predicted_filepath, expected_filepath, compared_filepath)
    


# Files to process
root_folder = Path(__file__).parents[1]
data_directory = root_folder / "data/"
filegroups = grouped_files(data_directory)

for file_root in filegroups:
    if file_root[0] == ".":
        continue
    files = filegroups[file_root]
    workflow(files, file_root, root_folder)

# For unit testing
# files = ["p2-l1-s0-r1.pkl"]
# file_root = files[0].split("-r")[0]
# workflow(files, file_root, root_folder)
