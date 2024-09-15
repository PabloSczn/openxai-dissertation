import numpy as np
import pandas as pd
from openxai.dgp_synthetic import generate_gaussians

# Initialize the synthetic data generator
generator = generate_gaussians(
    n_samples=10000,   # Total number of samples
    dimensions=20,     # Number of features
    n_clusters=100,    # Number of clusters (for generating complex data)
    distance_to_center=5,  # Distance between clusters
    upper_weight=1,    # Upper bound for feature weights
    lower_weight=-1,   # Lower bound for feature weights
    test_size=0.25,    # Split between training and testing
    seed=42            # Random seed for reproducibility
)

# Step 1: Generate cluster centers (mus)
cluster_centers = generator._get_mus()

# Step 2: Sample data around each cluster center (iterate over clusters)
X = []
y = []
for center in cluster_centers:
    X_cluster = np.random.multivariate_normal(center, generator.sigma, size=generator.n_samples // len(cluster_centers))
    y_cluster = (X_cluster @ generator.w > 0).astype(int)  # Linear decision boundary based on weights
    X.append(X_cluster)
    y.append(y_cluster)

# Step 3: Concatenate data from all clusters
X = np.vstack(X)
y = np.concatenate(y)

# Step 4: Split data into train and test sets
split_index = int(X.shape[0] * (1 - generator.test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Ground truth feature importance
ground_truth_weights = generator.w

# Convert the data into DataFrames
train_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
train_data['target'] = y_train

test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
test_data['target'] = y_test

# Save the train and test sets to CSV
train_data.to_csv('data/synthetic_train_data.csv', index=False)
test_data.to_csv('data/synthetic_test_data.csv', index=False)

# Save the ground truth feature importance weights
np.savetxt('data/ground_truth_weights.csv', ground_truth_weights, delimiter=',')

# Print confirmation
print("Datasets saved as CSV files.")
print("Ground truth weights saved as 'ground_truth_weights.csv'")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
