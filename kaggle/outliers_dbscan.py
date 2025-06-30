import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler # Important for distance-based algorithms
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('../data/train.csv')
df2 = pd.read_csv('../data/Fertilizer_Prediction.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df.drop(columns=['id'])

print("Original DataFrame shape:", df.shape)
print("Original DataFrame head:\n", df.head())

# 2. Select numerical features for outlier detection
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
X_numerical = df[numerical_cols]

# IMPORTANT: Scale the data before applying DBSCAN.
# DBSCAN is sensitive to the scale of features because it's distance-based.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols)

# 3. Initialize and fit the DBSCAN model
# eps (epsilon): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
# Tuning these parameters is crucial for DBSCAN.
# For this small dataset, these values are chosen for demonstration;
# for real data, you'd typically use techniques like a k-distance graph to find a good 'eps'.
dbscan = DBSCAN(eps=0.8, min_samples=3) # Example parameters
clusters = dbscan.fit_predict(X_scaled_df)

# Add cluster labels to the original DataFrame
df['cluster_label'] = clusters

# Separate inliers (points belonging to a cluster, label >= 0) and outliers (noise points, label == -1)
outliers_df = df[df['cluster_label'] == -1]
inliers_df = df[df['cluster_label'] >= 0] # Inliers will have cluster labels 0, 1, 2, etc.

print(f"\nNumber of data points identified as outliers: {len(outliers_df)}")
print(f"Number of data points identified as inliers: {len(inliers_df)}")
print(f"Number of clusters found: {df['cluster_label'].nunique() - (1 if -1 in df['cluster_label'].values else 0)}")


print("\nIdentified Outliers (cluster_label = -1):\n", outliers_df)

# 5. Remove outliers to create a cleaned dataset
df_cleaned = df[df['cluster_label'] != -1].drop(columns=['cluster_label'])

print("\nCleaned DataFrame shape:", df_cleaned.shape)
print("Cleaned DataFrame head:\n", df_cleaned.head())

# --- Visualization (Optional) ---
print("\nGenerating pairplot to visualize outliers and clusters...")
df_plot = df.copy()
# This column will have 'Inlier' and potentially 'Outlier'
df_plot['is_outlier'] = df_plot['cluster_label'].map(lambda x: 'Outlier' if x == -1 else 'Inlier')

# Define the order for consistent plotting
hue_order = ['Inlier', 'Outlier']

# Use seaborn pairplot to visualize relationships and highlight outliers.
# The fix is to use a dictionary for the `markers` argument.
sns.pairplot(
    df_plot,
    vars=numerical_cols,
    hue='is_outlier',
    hue_order=hue_order,  # Explicitly set the order of categories
    palette={'Inlier': 'blue', 'Outlier': 'red'},
    markers={'Inlier': 'o', 'Outlier': 'X'}  # Use a dictionary for markers
)
plt.suptitle('Outlier Detection using DBSCAN (Numerical Features)', y=1.02)
plt.show()

# The rest of your visualization code is great and doesn't need changes.
if len(numerical_cols) >= 2:
    # ... (your second plot code) ...df_plot = df.copy()
    df_plot['is_outlier'] = df_plot['cluster_label'].map(lambda x: 'Outlier' if x == -1 else 'Inlier')

    # Use seaborn pairplot to visualize relationships between numerical features
    # and highlight outliers.
    sns.pairplot(df_plot, vars=numerical_cols, hue='is_outlier', palette={'Inlier': 'blue', 'Outlier': 'red'}, markers=["o", "X"])
    plt.suptitle('Outlier Detection using DBSCAN (Numerical Features)', y=1.02)
    plt.show()
