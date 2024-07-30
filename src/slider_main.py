import json
import math
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import os
from data_visualization import combine_videos, visualize_slider, data_visualization
from ml_models import first_split, lstm_model
from data_processing import process_data, read_pkl_file, save_list_to_json_file, grouped_files
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
    filename = file_root + ".mp4"
    filepath = root_folder / ("visualizations/" + filename)
    data_visualization(mphands_data, file_to_save=filepath, preview=False) # file_to_save=filepath
    #######

    # Calculate slider distances from actual data    
    actual_distances = []
    for row in mphands_data:
        li = []
        point1 = {"x": row[fingerA * 3], "y": row[fingerA * 3+1], "z": row[fingerA * 3 +2]}
        for keypoint in fingerB:
            point2 = {"x": row[keypoint * 3], "y": row[keypoint * 3+1], "z": row[keypoint * 3 + 2]}
            li.append(distance(point1, point2))
        actual_distances.append(li)
        
    # Store actual distances
    filepath = root_folder / ("slider_data/actual_distance/" + file_root + ".json")
    with open(filepath, 'w') as f:
        json.dump(actual_distances, f)

    # Visualize actual distances + slider model
    #################    
    model_filepath = root_folder / ("slider_data/actual_distance/mapped_" + file_root + ".mp4")
    visualize_slider(np.array(actual_distances), file_to_save = model_filepath);
    
    visualization_filepath = root_folder / ("visualizations/" + file_root + ".mp4")
    
    compared_filepath = root_folder / ("slider_data/actual_distance/compared_" + file_root + ".mp4")
    combine_videos(visualization_filepath, model_filepath, compared_filepath)
    #################


    # Start ML models
    
    # Split data to test - train
    X_train, X_test, y_train, y_test = first_split(eit_data, mphands_data)
    
    size = X_train.shape[1]
    X_train_reshaped = X_train.reshape(-1, 1, size)
    X_test_reshaped = X_test.reshape(-1, 1, size)
    
    # Data visualization of test cases
    expected_filepath = root_folder / ("visualizations/" + file_root + "_actual_test.mp4")
    data_visualization(y_test, expected_filepath, preview=False, title="Ground Truth Hand")
    
    # Calculate distances for y_train and y_test
    y_train_distances = []
    for row in y_train:
        li = []
        point1 = {"x": row[fingerA * 3], "y": row[fingerA * 3+1], "z": row[fingerA * 3 +2]}
        for keypoint in fingerB:
            point2 = {"x": row[keypoint * 3], "y": row[keypoint * 3+1], "z": row[keypoint * 3 + 2]}
            li.append(distance(point1, point2))
        y_train_distances.append(li)
    y_train_distances = np.array(y_train_distances)

    y_test_distances = []
    for row in y_test:
        li = []
        point1 = {"x": row[fingerA * 3], "y": row[fingerA * 3+1], "z": row[fingerA * 3 +2]}
        for keypoint in fingerB:
            point2 = {"x": row[keypoint * 3], "y": row[keypoint * 3+1], "z": row[keypoint * 3 + 2]}
            li.append(distance(point1, point2))
        y_test_distances.append(li)
    y_test_distances = np.array(y_test_distances)
    
    # Data visualization of slider for test cases
    model_filepath = root_folder / ("visualizations/" + file_root + "_actual_slider_test.mp4")
    visualize_slider(np.array(y_test_distances), file_to_save = model_filepath, title="Slider from Ground Truth")

    # Save actual distances of test
    filepath = root_folder / ("slider_data/actual_distances_test/" + file_root + ".json")
    with open(filepath, 'w') as f:
        json.dump(y_test_distances.tolist(), f)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(1, size), return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.Dense(4)  # Output layer with 4 neurons
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_reshaped, y_train_distances, epochs=100, batch_size=32, validation_split=0.2)
    loss = model.evaluate(X_test_reshaped, y_test_distances)
    print("Test Loss for distance prediction from eit data:", loss)
    y_pred_distances = model.predict(X_test_reshaped)

    # Get errors for this model
    difference = y_test_distances - y_pred_distances
    difference_reshaped = difference.reshape(-1, 4, 1)
    mse_per_point = np.mean(np.square(difference_reshaped), axis=2)
    average_mses = np.mean(mse_per_point, axis=0)

    for index, mse in enumerate(average_mses):
        print("Distance of Point " + str(index) + " from Actual Averaged: " + str(mse))

    filepath = root_folder / ("slider_data/predicted_from_only_distances/errors/" + file_root + ".json")
    save_list_to_json_file(average_mses.tolist(), filepath)

    filepath = root_folder / ("slider_data/predicted_from_only_distances/" + file_root + ".json")
    with open(filepath, 'w') as f:
        json.dump(y_pred_distances.tolist(), f)

    # Data visualization of slider for predicting only distances
    model_filepath = root_folder / ("visualizations/" + file_root + "_from_distances_slider.mp4")
    visualize_slider(y_pred_distances, file_to_save = model_filepath, title="Slider from Predicted Distances")

    # Predict distances from modeling full 21 keypoints
    y_pred_visualization, overall_loss, point_errors = lstm_model(X_train, X_test, y_train, y_test)
    
    distances = []
    for row in y_pred_visualization:
        li = []
        point1 = {"x": row[fingerA * 3], "y": row[fingerA * 3+1], "z": row[fingerA * 3 +2]}
        for keypoint in fingerB:
            point2 = {"x": row[keypoint * 3], "y": row[keypoint * 3+1], "z": row[keypoint * 3 +2]}
            li.append(distance(point1, point2))
        distances.append(li)

    difference = y_test_distances - np.array(distances)
    difference_reshaped = difference.reshape(-1, 4, 1)
    mse_per_point = np.mean(np.square(difference_reshaped), axis=2)
    average_mses = np.mean(mse_per_point, axis=0)

    for index, mse in enumerate(average_mses):
        print("Distance of Point " + str(index) + " from visualization: " + str(mse))

    # Save errors for modeling of full 21 keypoints
    filepath = root_folder / ("slider_data/predicted_from_whole_visualization/errors/" + filename + ".json")
    save_list_to_json_file(average_mses.tolist(), filepath)
    
    # Save distances from modeling of full 21 keypoints
    filepath = root_folder / ("slider_data/predicted_from_whole_visualization/" + filename + ".json")
    with open(filepath, 'w') as file:
        json.dump(distances, file)
    
    # Data visualization of slider for predicting only distances
    model_filepath = root_folder / ("visualizations/" + file_root + "_from_21keypoints_slider.mp4")
    visualize_slider(np.array(distances), file_to_save = model_filepath, title="Slider from Predicted 21 Keypoints")

    # Do full visualization
    slider_1 = root_folder / ("visualizations/" + file_root + "_from_distances_slider.mp4")
    slider_2 = root_folder / ("visualizations/" + file_root + "_from_21keypoints_slider.mp4")
    slider_predicts = root_folder / ("visualizations/" + file_root + "_slider_predicts.mp4")
    combine_videos(slider_1, slider_2, slider_predicts)
    
    ground_1 = root_folder / ("visualizations/" + file_root + "_actual_test.mp4")
    ground_2 = root_folder / ("visualizations/" + file_root + "_actual_slider_test.mp4")
    ground_truths = root_folder / ("visualizations/" + file_root + "_ground_truths.mp4")
    combine_videos(ground_1, ground_2, ground_truths)

    compared_filepath = root_folder / ("slider_data/full_visualization/" + file_root + ".mp4")
    combine_videos(ground_truths, slider_predicts, compared_filepath)

    
fingerA = 4 # thumb
fingerB = [5, 6, 7, 8] # index
keyword = "slider"

def distance(point1, point2):
    return math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2 + (point1["z"] - point2["z"])**2)

# Files to process
root_folder = Path(__file__).parents[1]
data_directory = root_folder / "data/"
filegroups = grouped_files(data_directory)

for file_root in filegroups:
    if file_root[0] == ".":
        continue
    if (keyword in file_root):
        files = filegroups[file_root]
        workflow(files, file_root, root_folder)
