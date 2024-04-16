import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split

def random_split(X, y, random_state = 42):
    # random_state is for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def first_split(X, y, perc = 0.8):
    # First N% train, Remaining data as test 
    # For 80% with two data sets, around all of r2 is train, 60% of r3 is train, 40% of r3 is test

    # Determine the index to split the data
    split_index = int(perc * len(X))

    # Split the data into train and test sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def lstm_model(X_train, X_test, y_train, y_test):
    # Reshape data for LSTM input
    size = X_train.shape[1]
    X_train_reshaped = X_train.reshape(-1, 1, size)
    X_test_reshaped = X_test.reshape(-1, 1, size)

    # Define LSTM model with dropout regularization
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(1, size), return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
        tf.keras.layers.Dense(63)  # Output layer with 63 neurons
    ])

    # Consider changing the model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.LSTM(128, input_shape=(8, 128), return_sequences=True),
    #     tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
    #     tf.keras.layers.LSTM(128),
    #     tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
    #     tf.keras.layers.Dense(21 * 3)  # Output layer with 21 * 3 = 63 neurons
    # ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Calculate MSE
    loss = model.evaluate(X_test_reshaped, y_test)
    print("Test Loss:", loss)

    # Predict using the trained model
    y_pred = model.predict(X_test_reshaped)

    # Calculate the difference between y_pred and y_test
    difference = y_test - y_pred

    # Reshape difference
    difference_reshaped = difference.reshape(-1, 21, 3)

    # Compute the mean squared error along axis 2 (grouped by points in hand)
    mse_per_point = np.mean(np.square(difference_reshaped), axis=2)

    # Compute the average MSE across the test data samples
    average_mses = np.mean(mse_per_point, axis=0)

    for index, mse in enumerate(average_mses):
        print("Distance of Point " + str(index) + " from Actual Averaged: " + str(mse))

    return y_pred, loss, average_mses