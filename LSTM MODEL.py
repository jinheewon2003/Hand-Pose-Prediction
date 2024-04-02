import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

# Function to read pickle file
def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load data
pkl_file_path = 'adj_p15-l2-s3-r2.pkl'
pkl2_file_path = 'adj_p15-l2-s3-r3.pkl'
df = read_pkl_file(pkl_file_path)
df2 = read_pkl_file(pkl2_file_path)

# Extract features and targets
X1 = np.array(df['eit_data'].tolist())
X2 = np.array(df2['eit_data'].tolist())
X = np.concatenate((X1, X2), axis=0)

y1 = np.array(df['mphands_data'].tolist())
y2 = np.array(df2['mphands_data'].tolist())
y = np.concatenate((y1, y2), axis=0)

# # Scale data by 10 s.t. MSE is more accurate
# X = 10 * X
# y = 1000 * y

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM input (assuming each sequence has 1024 features)
X_train_reshaped = X_train_scaled.reshape(-1, 1, 1024)
X_test_reshaped = X_test_scaled.reshape(-1, 1, 1024)

# Define LSTM model with dropout regularization
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(1, 1024), return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
    tf.keras.layers.Dense(63)  # Output layer with 63 neurons
])

# # Define LSTM model with dropout regularization
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(128, input_shape=(8, 128), return_sequences=True),
#     tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
#     tf.keras.layers.LSTM(128),
#     tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
#     tf.keras.layers.Dense(21 * 3)  # Output layer with 21 * 3 = 63 neurons
# ])

# # Reshape data for LSTM input (assuming each sequence has 8 samples with 128 features each)
# X_train_reshaped = X_train_scaled.reshape(-1, 8, 128)
# X_test_reshaped = X_test_scaled.reshape(-1, 8, 128)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

# MSE does not work very well on the dataset
loss = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)

# Predict using the trained model
y_pred = model.predict(X_test_reshaped)

# Calculate the difference between y_pred and y_test
difference = y_test - y_pred

# Convert the difference array to a list (JSON doesn't support numpy arrays)
difference_list = difference.tolist()

# Save the difference to a new JSON file
difference_json_file_path = 'difference.json'
with open(difference_json_file_path, 'w') as f:
    json.dump(difference_list, f)
# Visualize the predictions and actual values side by side
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

special_connections_red = [
    (0, 1), (0, 5), (0, 17)
]
special_connections_purple = [
    (5, 9), (9, 13), (13, 17)
]

color_special_red = 'red'
color_special_purple = 'purple'
color_normal = 'blue'

data_pred = y_pred.reshape(-1, 21, 3)
data_test = y_test.reshape(-1, 21, 3)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
ax_pred, ax_test = axes

ax_pred.view_init(elev=20, azim=45)
ax_test.view_init(elev=20, azim=45)

lines_pred = [ax_pred.plot([], [], [])[0] for _ in range(len(connections))]
lines_test = [ax_test.plot([], [], [])[0] for _ in range(len(connections))]

def init():
    for line_pred, line_test in zip(lines_pred, lines_test):
        line_pred.set_data([], [])
        line_pred.set_3d_properties([])
        line_test.set_data([], [])
        line_test.set_3d_properties([])
    return lines_pred + lines_test

def update(frame):
    for i, connection in enumerate(connections):
        start_pred = data_pred[frame, connection[0], :]
        end_pred = data_pred[frame, connection[1], :]
        start_test = data_test[frame, connection[0], :]
        end_test = data_test[frame, connection[1], :]
        x_pred, y_pred, z_pred = zip(start_pred, end_pred)
        x_test, y_test, z_test = zip(start_test, end_test)
        if connection in special_connections_red:
            color = color_special_red
        elif connection in special_connections_purple:
            color = color_special_purple
        else:
            color = color_normal
        lines_pred[i].set_data(x_pred, y_pred)
        lines_pred[i].set_3d_properties(z_pred)
        lines_pred[i].set_color(color)
        lines_test[i].set_data(x_test, y_test)
        lines_test[i].set_3d_properties(z_test)
        lines_test[i].set_color('green')  # You can choose a different color for the test data
    
    # Find the limits for x, y, and z coordinates
    x_pred = data_pred[frame, :, 0]
    y_pred = data_pred[frame, :, 1]
    z_pred = data_pred[frame, :, 2]
    x_test = data_test[frame, :, 0]
    y_test = data_test[frame, :, 1]
    z_test = data_test[frame, :, 2]
    x_center_pred = np.mean(x_pred)
    y_center_pred = np.mean(y_pred)
    z_center_pred = np.mean(z_pred)
    max_range_pred = np.array([x_pred.max()-x_pred.min(), y_pred.max()-y_pred.min(), z_pred.max()-z_pred.min()]).max() / 2
    ax_pred.set_xlim(x_center_pred - max_range_pred, x_center_pred + max_range_pred)
    ax_pred.set_ylim(y_center_pred - max_range_pred, y_center_pred + max_range_pred)
    ax_pred.set_zlim(z_center_pred - max_range_pred, z_center_pred + max_range_pred)
    
    x_center_test = np.mean(x_test)
    y_center_test = np.mean(y_test)
    z_center_test = np.mean(z_test)
    max_range_test = np.array([x_test.max()-x_test.min(), y_test.max()-y_test.min(), z_test.max()-z_test.min()]).max() / 2
    ax_test.set_xlim(x_center_test - max_range_test, x_center_test + max_range_test)
    ax_test.set_ylim(y_center_test - max_range_test, y_center_test + max_range_test)
    ax_test.set_zlim(z_center_test - max_range_test, z_center_test + max_range_test)
    
    return lines_pred + lines_test

# Assuming the number of frames is the number of timestamps in the data
num_frames = data_pred.shape[0]

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=1000)

# Save the animation as a video file
video_file = 'predicted_random_animation.mp4'  # Provide the file name with extension (e.g., 'animation.mp4')
ani.save(video_file, writer='ffmpeg')

# Display the animation
plt.show()