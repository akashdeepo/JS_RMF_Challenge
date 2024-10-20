# Import required libraries
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
from scipy.stats import randint

# Create a metrics folder to save logs
metrics_dir = "metrics"
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

print("Starting data loading...")

# Step 1: Load Data
# Define the path to your data
train_path = r"your_data_directory_here/train.parquet"  # Update the path

# Get all partition files
train_files = glob(os.path.join(train_path, 'partition_id=*/*.parquet'))

# Load the data
data_list = []
for file in train_files:
    print(f"Loading data from: {file}")
    data = pd.read_parquet(file)
    data_list.append(data)

# Concatenate all partitions into one DataFrame
sample_partition = pd.concat(data_list, ignore_index=True)
print(f"Data loaded. Total records: {len(sample_partition)}")

# Step 2: Handle Missing Values
print("Handling missing values...")
# Impute missing values with median for moderate missingness
sample_partition.fillna(sample_partition.median(), inplace=True)

# Drop columns with more than 80% missing data
threshold = 0.8 * len(sample_partition)
sample_partition = sample_partition.dropna(thresh=threshold, axis=1)

# Log the shape of the dataset after missing value handling
with open(os.path.join(metrics_dir, "updated_dataset_shape.txt"), "w") as f:
    f.write(f"Shape after handling missing values: {sample_partition.shape}\n")

print(f"Shape after missing value handling: {sample_partition.shape}")

# Step 3: Feature Engineering (Lagged Features)
print("Creating lagged features for responder_6...")
# Create lagged features for responder_6 (1-day lag)
sample_partition['responder_6_lag_1'] = sample_partition['responder_6'].shift(1)

# Drop any rows with NaNs generated from the lagging
sample_partition.dropna(inplace=True)

# Log the lagged features creation
with open(os.path.join(metrics_dir, "lagged_features_log.txt"), "w") as f:
    f.write("Created lagged features for responder_6 (1-day lag)\n")

print(f"Shape after creating lagged features: {sample_partition.shape}")

# Step 4: Split the Data into Features and Target
print("Splitting data into features and target...")
# Define target and features
X = sample_partition.drop(columns=['responder_6'])  # Features
y = sample_partition['responder_6']                # Target

# Step 5: Split the data into training and test sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log the split sizes
with open(os.path.join(metrics_dir, "data_split_log.txt"), "w") as f:
    f.write(f"Training data shape: {X_train.shape}\n")
    f.write(f"Test data shape: {X_test.shape}\n")

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Step 6: Hyperparameter Tuning using RandomizedSearchCV
print("Starting hyperparameter tuning...")

# Define hyperparameters to tune
param_dist = {
    'n_estimators': randint(50, 200),  # Number of trees
    'max_depth': randint(5, 20),       # Maximum depth of the trees
    'min_samples_split': randint(2, 20)  # Minimum number of samples required to split a node
}

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_
print(f"Best hyperparameters found: {best_params}")

# Log the best hyperparameters
with open(os.path.join(metrics_dir, "best_hyperparameters.txt"), "w") as f:
    f.write(f"Best hyperparameters: {best_params}\n")

# Step 7: Train Random Forest Model with Best Hyperparameters
print("Training the model with best hyperparameters...")
rf_best_model = random_search.best_estimator_
rf_best_model.fit(X_train, y_train)

# Step 8: Make Predictions on Test Data
print("Making predictions on test data...")
y_pred = rf_best_model.predict(X_test)

# Calculate R² score
r2_score_value = r2_score(y_test, y_pred)
print(f"R² score: {r2_score_value}")

# Log R² score
with open(os.path.join(metrics_dir, "random_forest_r2_score.txt"), "w") as f:
    f.write(f"Random Forest R² score: {r2_score_value}\n")

# Step 9: Calculate Weighted R² Score
print("Calculating weighted R² score...")

def weighted_r2_score(y_true, y_pred, sample_weights):
    # Calculate weighted squared error (numerator)
    weighted_squared_error = np.sum(sample_weights * (y_true - y_pred) ** 2)
    
    # Calculate weighted total sum of squares (denominator)
    weighted_total_sum_of_squares = np.sum(sample_weights * (y_true ** 2))
    
    # Calculate weighted R² score
    weighted_r2 = 1 - (weighted_squared_error / weighted_total_sum_of_squares)
    return weighted_r2

# Assuming the 'weight' column is present in X_test
sample_weights = X_test['weight']  # Ensure 'weight' is included

# Calculate weighted R² score
weighted_r2 = weighted_r2_score(y_test, y_pred, sample_weights)
print(f"Weighted R² score: {weighted_r2}")

# Log the weighted R² score
with open(os.path.join(metrics_dir, "weighted_r2_score.txt"), "w") as f:
    f.write(f"Weighted R² score: {weighted_r2}\n")

# Step 10: Save the Model for Submission on Kaggle
print("Saving the model...")
joblib.dump(rf_best_model, 'random_forest_best_model.pkl')

# Log model save status
with open(os.path.join(metrics_dir, "model_save_log.txt"), "w") as f:
    f.write("Random Forest model saved as random_forest_best_model.pkl\n")

print("Model training completed and saved.")
