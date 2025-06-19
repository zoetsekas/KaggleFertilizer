import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from torch import nn


class FertilizerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, l1_units, l2_units, l3_units, dropout_rate, activation_fn_name):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_units)
        self.bn1 = nn.BatchNorm1d(l1_units) # Good practice to include BatchNorm

        self.fc2 = nn.Linear(l1_units, l2_units)
        self.bn2 = nn.BatchNorm1d(l2_units)

        self.fc3 = nn.Linear(l2_units, l3_units)
        self.bn3 = nn.BatchNorm1d(l3_units)

        self.fc4 = nn.Linear(l3_units, num_classes) # Final layer

        self.dropout = nn.Dropout(dropout_rate)

        # Select activation function based on string name
        if activation_fn_name == "relu":
            self.activation = nn.ReLU()
        elif activation_fn_name == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_fn_name == "elu":
            self.activation = nn.ELU()
        elif activation_fn_name == "gelu":
            self.activation = nn.GELU()
        elif activation_fn_name == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn_name}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc4(x) # Output logits
        return x

def round_to_half(x):
    return round(x * 2) / 2

def divide_columns(df, numerator_col, denominator_col, result_col):
    """Divides two columns, handling zero denominators.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerator_col (str): The name of the column to be divided.
        denominator_col (str): The name of the column to divide by.
        result_col (str): The name of the new column to store the result.

    Returns:
        pd.DataFrame: The DataFrame with the new column.
    """
    df[result_col] = df[denominator_col].replace(0, 1)
    df[result_col] = df[numerator_col] / df[result_col]
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Nutrient Ratios and Totals ---
    # Add a small epsilon to avoid division by zero
    df = divide_columns(numerator_col='Nitrogen', denominator_col='Phosphorous', result_col='NP_Ratio', df=df)
    df['NP_Ratio_equal_width'] = pd.cut(df['NP_Ratio'], bins=4, labels=False)

    df = divide_columns(numerator_col='Nitrogen', denominator_col='Potassium', result_col='NK_Ratio', df=df)
    df['NK_Ratio_equal_width'] = pd.cut(df['NK_Ratio'], bins=4, labels=False)

    df = divide_columns(numerator_col='Phosphorous', denominator_col='Potassium', result_col='PK_Ratio', df=df)
    df['PK_Ratio_equal_width'] = pd.cut(df['PK_Ratio'], bins=4, labels=False)

    df['Total_NPK'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']

    # Handle cases where Total_NPK might be zero for percentage calculation
    total_npk_safe = df['Total_NPK'].replace(0, 1)
    df['N_Percentage'] = round_to_half((df['Nitrogen'] / total_npk_safe) * 100)
    df['N_Percentage_equal_width'] = pd.cut(df['N_Percentage'], bins=4, labels=False)

    df['P_Percentage'] = round_to_half((df['Phosphorous'] / total_npk_safe) * 100)
    df['P_Percentage_equal_width'] = pd.cut(df['P_Percentage'], bins=4, labels=False)

    df['K_Percentage'] = round_to_half((df['Potassium'] / total_npk_safe) * 100)
    df['K_Percentage_equal_width'] = pd.cut(df['K_Percentage'], bins=4, labels=False)

    # --- 2. Environmental Interactions ---
    df['Temp_Moisture_Interaction'] = round_to_half((df['Temparature'] * df['Moisture']) / 100)
    df['Temp_Moisture_Interaction_equal_width'] = pd.cut(df['Temp_Moisture_Interaction'], bins=4, labels=False)

    df['Temp_Humidity_Interaction'] = round_to_half((df['Temparature'] * df['Moisture']) / 100)
    df['Temp_Humidity_Interaction_equal_width'] = pd.cut(df['Temp_Humidity_Interaction'], bins=4, labels=False)

    df['Hum_Moisture_Interaction'] = round_to_half((df['Humidity'] * df['Moisture']) / 100)
    df['Hum_Moisture_Interaction_equal_width'] = pd.cut(df['Hum_Moisture_Interaction'], bins=4, labels=False)


    # --- 3. Polynomial Features ---
    df['Temperature_Squared'] = round_to_half((df['Temparature'] ** 2) / 100)
    df['Temperature_Squared_equal_width'] = pd.cut(df['Temperature_Squared'], bins=4, labels=False)

    df['Humidity_Squared'] = round_to_half((df['Humidity'] ** 2) / 100)
    df['Humidity_Squared_equal_width'] = pd.cut(df['Humidity_Squared'], bins=4, labels=False)

    df['Moisture_Squared'] = round_to_half((df['Moisture'] ** 2) / 100)
    df['Moisture_Squaredequal_width'] = pd.cut(df['Moisture_Squared'], bins=4, labels=False)

    # --- 4. Binning ---
    # Temperature Binning
    bins_temparature = [25, 30, 35, df['Temparature'].max() + 2]  # +1 to ensure the max value is included
    labels_temparature = ['Low', 'Medium', 'High']

    df['Temparature_Binned'] = pd.cut(df['Temparature'], bins=bins_temparature, labels=labels_temparature,
                                   right=False)
    # Humidity Binning
    bins_humidity = [50, 60, 70, df['Humidity'].max() + 8]  # +1 to ensure the max value is included
    labels_humidity = ['Low', 'Medium', 'High']

    df['Humidity_Binned'] = pd.cut(df['Humidity'], bins=bins_humidity, labels=labels_humidity,
                                      right=False)

    # Moisture Binning
    bins_moisture = [25, 35, 45, 55, df['Moisture'].max() + 2]  # +1 to ensure the max value is included
    labels_moisture = ['Low', 'Medium', 'High', 'Extreme']

    df['Moisture_Binned'] = pd.cut(df['Moisture'], bins=bins_moisture, labels=labels_moisture,
                                      right=False)

    # Nitrogen Binning
    bins_nitrogen = [0, 15, 30, df['Nitrogen'].max() + 3]  # +1 to ensure the max value is included
    labels_nitrogen = ['Low', 'Medium', 'High']

    df['Nitrogen_Binned'] = pd.cut(df['Nitrogen'], bins=bins_nitrogen, labels=labels_nitrogen,
                                          right=False)  # right=False for [low, high) intervals

    # Potassium Binning
    bins_potassium = [0, 5, 10, 15, df['Potassium'].max() + 1]  # +1 to ensure the max value is included
    labels_potassium = ['Low', 'Medium', 'High', 'Extreme']

    df['Potassium_Binned'] = pd.cut(df['Potassium'], bins=bins_potassium, labels=labels_potassium,
                                   right=False)  # right=False for [low, high) intervals


    # Phosphorous Binning
    bins_phosphorous = [0, 10, 20, 30, df['Phosphorous'].max() + 3]  # +1 to ensure the max value is included
    labels_phosphorous = ['Low', 'Medium', 'High', 'Extreme']

    df['Phosphorous_Binned'] = pd.cut(df['Phosphorous'], bins=bins_phosphorous, labels=labels_phosphorous,
                                    right=False)  # right=False for [low, high) intervals

    return df

def feature_selection(input_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    # --- 3. Apply Filter Methods ---

    # a) Variance Threshold
    # This method removes features with variance below a certain threshold.
    # The default is to remove features with zero variance.
    var_selector = VarianceThreshold(threshold=0.01)
    x_var_filtered = var_selector.fit_transform(input_df)
    # Get the names of the features that were kept
    kept_features_var = input_df.columns[var_selector.get_support()]
    dropped_features_var = input_df.columns[~var_selector.get_support()]

    x_df_var_filtered = pd.DataFrame(x_var_filtered, columns=kept_features_var)

    print("--- Method 1: Variance Threshold ---")
    print(f"Original number of features: {input_df.shape[1]}")
    print(f"Features after Variance Threshold: {x_df_var_filtered.shape[1]}")
    print(f"Features dropped: {list(dropped_features_var)}")
    print(f"Shape after Variance Threshold: {x_df_var_filtered.shape}")
    print("\n" + "=" * 50 + "\n")

    # b) Drop Highly Correlated Features
    # Create a correlation matrix
    corr_matrix = x_df_var_filtered.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]

    x_df_corr_filtered = x_df_var_filtered.drop(to_drop_corr, axis=1)

    print("--- Method 2: Dropping Highly Correlated Features ---")
    print(f"Features before dropping correlated ones: {x_df_var_filtered.shape[1]}")
    print(f"Features after dropping correlated ones: {x_df_corr_filtered.shape[1]}")
    print(f"Features dropped: {to_drop_corr}")
    print(f"Shape after dropping correlated features: {x_df_corr_filtered.shape}")
    print("\n" + "=" * 50 + "\n")

    # --- 4. Apply Wrapper Method: Recursive Feature Elimination (RFE) ---
    # RFE works by recursively removing attributes and building a model on those attributes that remain.
    # It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
    # NOTE: We use the dataset after initial filtering for better performance.
    model_rfe = LogisticRegression(solver='liblinear')
    # Select the best 15 features
    rfe_selector = RFE(estimator=model_rfe, n_features_to_select=15, step=1)
    rfe_selector.fit(x_df_corr_filtered, target_df)

    kept_features_rfe = x_df_corr_filtered.columns[rfe_selector.get_support()]
    x_df_rfe_filtered = x_df_corr_filtered[kept_features_rfe]

    print("--- Method 3: Recursive Feature Elimination (RFE) ---")
    print(f"Number of features before RFE: {x_df_corr_filtered.shape[1]}")
    print(f"Number of features selected by RFE: {x_df_rfe_filtered.shape[1]}")
    print(f"Top 15 features selected by RFE: {list(kept_features_rfe)}")
    print("\n" + "=" * 50 + "\n")

    # --- 5. Apply Embedded Method: SelectFromModel with Random Forest ---
    # Embedded methods use algorithms that have built-in feature selection methods.
    # For instance, Lasso and RF have their own feature selection methods.
    # We will use RandomForestClassifier to select features.
    # NOTE: We use the dataset after initial filtering for better performance.
    model_embed = RandomForestClassifier(n_estimators=100, random_state=42)
    embed_selector = SelectFromModel(estimator=model_embed,
                                     threshold='median')  # drop features with importance below the median
    embed_selector.fit(x_df_corr_filtered, target_df)

    kept_features_embed = x_df_corr_filtered.columns[embed_selector.get_support()]
    x_df_embed_filtered = x_df_corr_filtered[kept_features_embed]

    print("--- Method 4: Embedded Method (Random Forest Importance) ---")
    print(f"Number of features before Embedded Selection: {x_df_corr_filtered.shape[1]}")
    print(f"Number of features selected by Embedded Method: {x_df_embed_filtered.shape[1]}")
    print(f"Features selected: {list(kept_features_embed)}")
    print("\n" + "=" * 50 + "\n")

    # --- Final Comparison ---
    print("--- Summary of Feature Dropping ---")
    print(f"Original number of features: {input_df.shape[1]}")
    print(f"Dropped by VarianceThreshold: {list(dropped_features_var)}")
    print(f"Dropped by High Correlation: {to_drop_corr}")
    print(f"Final feature count after initial filtering (Variance + Corr): {x_df_corr_filtered.shape[1]}")
    print(f"Final feature count after RFE: {x_df_rfe_filtered.shape[1]}")
    print(f"Final feature count after Embedded Selection: {x_df_embed_filtered.shape[1]}")

    return x_df_embed_filtered

