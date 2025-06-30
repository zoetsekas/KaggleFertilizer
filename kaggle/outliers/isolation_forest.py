import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

df1 = pd.read_csv('../../data/train.csv')
df2 = pd.read_csv('../../data/Fertilizer_Prediction.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df.drop(columns=['id'])

print("Original DataFrame shape:", df.shape)
print("Original DataFrame head:\n", df.head())

# 2. Select numerical features for outlier detection
# For simplicity, we'll use all numerical columns.
# If you want to include categorical features, you'd need to one-hot encode them first.
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
X_numerical = df[numerical_cols]

# 3. Initialize and fit the Isolation Forest model
# contamination: The estimated proportion of outliers in the dataset.
# Since this is a small dataset, a small contamination value is appropriate.
# Adjust this value based on your domain knowledge or by experimenting.
iso_forest = IsolationForest(contamination=0.1, random_state=42) # Assuming 10% outliers
iso_forest.fit(X_numerical)

# 4. Predict outlier labels
# predict() returns 1 for inliers and -1 for outliers
df['outlier_prediction'] = iso_forest.predict(X_numerical)

# Separate inliers and outliers
outliers_df = df[df['outlier_prediction'] == -1]
inliers_df = df[df['outlier_prediction'] == 1]

print(f"\nNumber of data points identified as outliers: {len(outliers_df)}")
print(f"Number of data points identified as inliers: {len(inliers_df)}")

print("\nIdentified Outliers:\n", outliers_df)

# 5. Remove outliers to create a cleaned dataset
df_cleaned = df[df['outlier_prediction'] == 1].drop(columns=['outlier_prediction'])

print("\nCleaned DataFrame shape:", df_cleaned.shape)
print("Cleaned DataFrame head:\n", df_cleaned.head())

# --- Visualization (Optional) ---
# For 2D visualization, we'll use a pairplot or select two features.
# A pairplot can be useful for small datasets to see relationships and outliers.
print("\nGenerating pairplot to visualize outliers...")
# Add the outlier_prediction back for visualization purposes
df_plot = df.copy()
df_plot['is_outlier'] = df_plot['outlier_prediction'].map({1: 'Inlier', -1: 'Outlier'})

# Use seaborn pairplot to visualize relationships between numerical features
# and highlight outliers. This can be slow for many features.
# For a quick view, you might just pick 2-3 key features.
sns.pairplot(df_plot, vars=numerical_cols, hue='is_outlier', palette={'Inlier': 'blue', 'Outlier': 'red'}, markers=["o", "X"])
plt.suptitle('Outlier Detection using Isolation Forest (Numerical Features)', y=1.02) # Adjust title position
plt.show()

# You can also visualize a specific 2D projection if you have many features
if len(numerical_cols) >= 2:
    plt.figure(figsize=(10, 7))
    plt.scatter(inliers_df[numerical_cols[0]], inliers_df[numerical_cols[1]], c='blue', label='Inliers', alpha=0.7)
    plt.scatter(outliers_df[numerical_cols[0]], outliers_df[numerical_cols[1]], c='red', marker='X', s=100, label='Outliers')
    plt.title(f'Outliers in {numerical_cols[0]} vs {numerical_cols[1]}')
    plt.xlabel(numerical_cols[0])
    plt.ylabel(numerical_cols[1])
    plt.legend()
    plt.grid(True)
    plt.show()