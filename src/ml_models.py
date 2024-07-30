import numpy as np
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

def random_split(X, y, random_state=42):
    # Randomly split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def first_split(X, y, perc=0.8):
    # First N% train, Remaining data as test
    split_index = floor(int((1-perc) * len(X)) / 2)
    half_index = int(0.5 * len(X))

    X_train = np.concatenate((X[:half_index - split_index], X[half_index + split_index:]), axis=0)
    X_test = X[half_index - split_index:half_index + split_index]
    y_train = np.concatenate((y[:half_index - split_index], y[half_index + split_index:]), axis=0)
    y_test = y[half_index - split_index:half_index + split_index]
    return X_train, X_test, y_train, y_test

def lstm_model(X_train, X_test, y_train, y_test):
    # Reshape data for LSTM input
    size = X_train.shape[1]
    X_train_reshaped = X_train.reshape(-1, 1, size)
    X_test_reshaped = X_test.reshape(-1, 1, size)

    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            inputs = keras.Input(shape=(1, size))
            x = layers.Bidirectional(layers.LSTM(
                units=hp.Int('units_1', min_value=64, max_value=256, step=32),
                return_sequences=True
            ))(inputs)
            x = layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1))(x)
            
            x = layers.Bidirectional(layers.LSTM(
                units=hp.Int('units_2', min_value=64, max_value=256, step=32),
                return_sequences=True
            ))(x)
            x = layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1))(x)
            
            x = layers.Bidirectional(layers.LSTM(
                units=hp.Int('units_3', min_value=32, max_value=128, step=32)
            ))(x)
            x = layers.Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1))(x)
            
            outputs = layers.Dense(63, activation='linear')(x)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
                ),
                loss='mse',
                metrics=['mae']
            )
            
            return model

    # Initialize the tuner
    tuner = kt.RandomSearch(
        MyHyperModel(),
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        directory='my_dir',
        project_name='lstm_tuning'
    )

    # Search for the best hyperparameters
    tuner.search(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ])

    # Get the best model
    model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model
    loss, mae = model.evaluate(X_test_reshaped, y_test)
    print("Test Loss (MSE):", loss)
    print("Test MAE:", mae)

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

# from math import floor
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from keras import layers
# from kerastuner import HyperModel, RandomSearch
# from kerastuner.engine.hyperparameters import HyperParameters

# def random_split(X, y, random_state = 42):
#     # random_state is for reproducibility
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
#     return X_train, X_test, y_train, y_test

# def first_split(X, y, perc = 0.8):
#     # First N% train, Remaining data as test 
#     # For 80% with two data sets, around all of r2 is train, 60% of r3 is train, 40% of r3 is test

#     # Determine the index to split the data
#     split_index = floor(int((1-perc) * len(X))/2)
#     half_index = int(0.5 * len(X))

#     # Split the data into train and test sets
#     X_train = np.concatenate((X[:half_index - split_index], X[half_index + split_index:]), axis=0)
#     X_test = X[half_index - split_index:half_index + split_index]
#     y_train = np.concatenate((y[:half_index - split_index], y[half_index + split_index:]), axis=0)
#     y_test = y[half_index - split_index:half_index + split_index]
#     return X_train, X_test, y_train, y_test

# def lstm_model(X_train, X_test, y_train, y_test):
#     # Reshape data for LSTM input
#     size = X_train.shape[1]
#     X_train_reshaped = X_train.reshape(-1, 1, size)
#     X_test_reshaped = X_test.reshape(-1, 1, size)

#     # # Define LSTM model with dropout regularization
#     # model = tf.keras.Sequential([
#     #     tf.keras.layers.LSTM(128, input_shape=(1, size), return_sequences=True),
#     #     tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
#     #     tf.keras.layers.LSTM(128),
#     #     tf.keras.layers.Dropout(0.2),  # Dropout layer with dropout rate of 0.2
#     #     tf.keras.layers.Dense(63)  # Output layer with 63 neurons
#     # ])

#     # # Compile the model
#     # model.compile(optimizer='adam', loss='mse')

#     # # Train the model
#     # model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)

#     # # Calculate MSE
#     # loss = model.evaluate(X_test_reshaped, y_test)
#     # print("Test Loss:", loss)

#     class MyHyperModel(HyperModel):
#         def build(self, hp):
#             model = tf.keras.Sequential()
            
#             # First LSTM layer
#             model.add(layers.Bidirectional(layers.LSTM(
#                 units=hp.Int('units_1', min_value=64, max_value=256, step=32),
#                 input_shape=(1, size),
#                 return_sequences=True
#             )))
#             model.add(layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
            
#             # Second LSTM layer
#             model.add(layers.Bidirectional(layers.LSTM(
#                 units=hp.Int('units_2', min_value=64, max_value=256, step=32),
#                 return_sequences=True
#             )))
#             model.add(layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
            
#             # Third LSTM layer
#             model.add(layers.Bidirectional(layers.LSTM(
#                 units=hp.Int('units_3', min_value=32, max_value=128, step=32)
#             )))
#             model.add(layers.Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))
            
#             # Output layer
#             model.add(layers.Dense(63, activation='linear'))
            
#             # Compile model
#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(
#                     hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
#                 ),
#                 loss='mse',
#                 metrics=['mae']
#             )
            
#             return model

#     # Initialize the tuner
#     tuner = RandomSearch(
#         MyHyperModel(),
#         objective='val_loss',
#         max_trials=20,
#         executions_per_trial=1,
#         directory='my_dir',
#         project_name='lstm_tuning'
#     )

#     # Search for the best hyperparameters
#     tuner.search(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[
#         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     ])

#     # Get the best model
#     model = tuner.get_best_models(num_models=1)[0]

#     # Evaluate the best model
#     loss, mae = model.evaluate(X_test_reshaped, y_test)
#     print("Test Loss (MSE):", loss)
#     print("Test MAE:", mae)

#     # Predict using the trained model
#     y_pred = model.predict(X_test_reshaped)

#     # Calculate the difference between y_pred and y_test
#     difference = y_test - y_pred

#     # Reshape difference
#     difference_reshaped = difference.reshape(-1, 21, 3)

#     # Compute the mean squared error along axis 2 (grouped by points in hand)
#     mse_per_point = np.mean(np.square(difference_reshaped), axis=2)

#     # Compute the average MSE across the test data samples
#     average_mses = np.mean(mse_per_point, axis=0)

#     for index, mse in enumerate(average_mses):
#         print("Distance of Point " + str(index) + " from Actual Averaged: " + str(mse))

#     return y_pred, loss, average_mses