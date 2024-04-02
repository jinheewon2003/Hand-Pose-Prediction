import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

pkl_file_path = 'adj_p15-l2-s3-r2.pkl'
pkl2_file_path = 'adj_p15-l2-s3-r3.pkl'

def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

df = read_pkl_file(pkl_file_path)
df2 = read_pkl_file(pkl2_file_path)

# Extracting features and targets
X = np.array(df['eit_data'].tolist())
y = np.array(df['mphands_data'].tolist())

# Creating the Ridge Regression model
alpha = 0.25  # Regularization parameter
model = Ridge(alpha=alpha)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive
mse_scores = -cv_scores

# Print the mean squared error scores
print("Mean Squared Error (with Ridge Regression) for each fold:")
for i, mse_score in enumerate(mse_scores):
    print(f"Fold {i+1}: {mse_score}")

# Calculate and print the mean MSE across all folds
mean_mse = np.mean(mse_scores)
print("\nMean Squared Error (with Ridge Regression) across all folds:", mean_mse)
