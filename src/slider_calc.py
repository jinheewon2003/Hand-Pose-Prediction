import json
import math
import pickle

import tensorflow as tf
from data_processing import read_pkl_file, save_list_to_json_file
import numpy as np
import pandas as pd
from pathlib import Path
import os
from data_visualization import combine_videos, visualize_slider

from ml_models import first_split, lstm_model


fingerA = 4 # thumb
fingerB = [5, 6, 7, 8] # index
keyword = "slider"

def distance(point1, point2):
    return math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2 + (point1["z"] - point2["z"])**2)

# Calculate distance between finger A to finger B actual
root_folder = Path(__file__).parents[1]
data_directory = root_folder / "adj_data/"
info = []
for filename in os.listdir(data_directory):
    if (filename[0] == "."):
        continue
    if (keyword in filename):
        info.append(filename)

for file in info:
    file_name = root_folder / ("adj_data/" + file)
    df = read_pkl_file(file_name)
    
    mphands_data = []
    mphands_data.extend(df['mphands_data'].tolist())
    
    distances = []
    for row in mphands_data:
        li = []
        point1 = {"x": row[fingerA * 3], "y": row[fingerA * 3+1], "z": row[fingerA * 3 +2]}
        for keypoint in fingerB:
            point2 = {"x": row[keypoint * 3], "y": row[keypoint * 3+1], "z": row[keypoint * 3 +2]}
            li.append(distance(point1, point2))
        distances.append(li)

    #################    
    model_filepath = root_folder / ("slider_data/actual_distance/mapped_" + file + ".mp4")
    visualize_slider(distances[:500], file_to_save = model_filepath);
    
    visualization_filepath = root_folder / ("visualizations/" + file + ".mp4")
    
    compared_filepath = root_folder / ("slider_data/actual_distance/compared_" + file + ".mp4")
    combine_videos(model_filepath, visualization_filepath, compared_filepath)
    #################
    
    filename = file
    filepath = root_folder / ("slider_data/actual_distance/" + filename + ".json")
    with open(filepath, 'w') as f:
        json.dump(distances, f)

    # Predict distance between just based on eit data
    eit_data = []
    eit_data.extend(df['eit_data'].tolist())
    
    distances = np.array(distances)
    eit_data = np.array(eit_data)
    
    # distances is y, eit_data is x
    X_train, X_test, y_train, y_test = first_split(eit_data, distances)
    size = X_train.shape[1]
    X_train_reshaped = X_train.reshape(-1, 1, size)
    X_test_reshaped = X_test.reshape(-1, 1, size)

    # Define LSTM model with dropout regularization
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(1, size), return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.Dense(4)  # Output layer with 4 neurons
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

    loss = model.evaluate(X_test_reshaped, y_test)
    print("Test Loss:", loss)

    y_pred = model.predict(X_test_reshaped)
    filename = file
    filepath = root_folder / ("slider_data/predicted_from_only_distances/" + filename + ".json")
    with open(filepath, 'w') as f:
        json.dump(y_pred.tolist(), f)
        
    filename = file
    filepath = root_folder / ("slider_data/actual_distances_test/" + filename + ".json")
    with open(filepath, 'w') as f:
        json.dump(y_test.tolist(), f)
    
# Calculate distance between finger A to finger B predicted

root_folder = Path(__file__).parents[1]
data_directory = root_folder / "predicted_ys/"
info = []
for filename in os.listdir(data_directory):
    if (filename[0] == "."):
        continue
    if (keyword in filename):
        info.append(filename)

for file in info:
    file_name = root_folder / ("predicted_ys/" + file)
    with open(file_name, 'r') as f:
        content = f.read()

    mphands_data = eval(content)
    
    distances = []
    for row in mphands_data:
        li = []
        point1 = {"x": row[fingerA * 3], "y": row[fingerA * 3+1], "z": row[fingerA * 3 +2]}
        for keypoint in fingerB:
            point2 = {"x": row[keypoint * 3], "y": row[keypoint * 3+1], "z": row[keypoint * 3 +2]}
            li.append(distance(point1, point2))
        distances.append(li)
    
    filename = file
    filepath = root_folder / ("slider_data/predicted_from_whole_visualization/" + filename + ".json")
    with open(filepath, 'w') as file:
        json.dump(distances, file)

