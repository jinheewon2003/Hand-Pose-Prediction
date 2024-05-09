# Data Processing and Analysis Workflow

This repository contains scripts for processing, analyzing, and visualizing motion capture data using Python. The workflow includes data preprocessing, model training, and visualization steps.

## Requirements

- Python 3
- NumPy
- Matplotlib
- TensorFlow
- scikit-learn

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/yourrepository.git
```
Navigate to the cloned directory:

```
cd yourrepository
```

## Usage

### Data Processing

The `data_processing.py` script contains functions for reading and processing motion capture data stored in pickle files. It includes functions such as:

- `read_pkl_file(file_path)`: Reads data from a pickle file.
- `save_list_to_json_file(lst, filename)`: Saves a list to a JSON file.
- `grouped_files(folder_path)`: Groups files based on a common root filename.

To process data, you can run the `data_processing.py` script directly or integrate its functions into your workflow.

### Data Visualization

The `data_visualization.py` script provides functions for visualizing motion capture data in 3D space. It uses Matplotlib for plotting and animation. Key functions include:

- `data_visualization(data, file_to_save=None, preview=False, speed=500)`: Creates a 3D animation of the motion capture data.
- `combine_videos(video1_path, video2_path, output_path)`: Combines two video files side by side using ffmpeg.

To visualize motion capture data, you can use the provided functions and customize the visualization parameters as needed.

### Machine Learning Models

The `ml_models.py` script contains functions for training machine learning models on motion capture data. It includes functions such as:

- `random_split(X, y, random_state=42)`: Splits data into training and testing sets randomly.
- `first_split(X, y, perc=0.8)`: Splits data into training and testing sets based on a specified percentage.
- `lstm_model(X_train, X_test, y_train, y_test)`: Trains an LSTM model on the data and evaluates its performance.

To train machine learning models on motion capture data, you can use the provided functions and adjust the model architecture and parameters as needed.

### Workflow Automation

The `main.py` script automates the entire data processing and analysis workflow. It reads motion capture data files, processes them, trains machine learning models, and generates visualizations. 

To use the workflow, simply run the `main.py` script and provide the necessary input parameters, such as file paths and model configurations.
