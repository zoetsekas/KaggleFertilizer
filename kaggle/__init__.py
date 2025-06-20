import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from torch import nn


class FertilizerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, l1_units, l2_units, l3_units, l4_units, dropout_rate, activation_fn_name):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_units)
        self.bn1 = nn.BatchNorm1d(l1_units)

        self.fc2 = nn.Linear(l1_units, l2_units)
        self.bn2 = nn.BatchNorm1d(l2_units)

        self.fc3 = nn.Linear(l2_units, l3_units)
        self.bn3 = nn.BatchNorm1d(l3_units)

        self.fc4 = nn.Linear(l3_units, l4_units)
        self.bn4 = nn.BatchNorm1d(l4_units)

        self.fc5 = nn.Linear(l4_units, num_classes) # Final layer

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

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc5(x) # Output logits
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
    """
    Adds several new features based on existing data to improve model performance.
    """
    # --- Interaction and Ratio Features ---
    df['NP_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['NK_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['PK_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
    df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']

    # --- Advanced Nutrient Balance Features ---
    df['N_percent_of_total'] = df['Nitrogen'] / (df['Total_NPK'] + 1e-6)
    df['P_percent_of_total'] = df['Phosphorous'] / (df['Total_NPK'] + 1e-6)
    df['K_percent_of_total'] = df['Potassium'] / (df['Total_NPK'] + 1e-6)
    df['npk_imbalance'] = df[['N_percent_of_total', 'P_percent_of_total', 'K_percent_of_total']].std(axis=1)

    # --- Environmental Stress Features ---
    # Vapor Pressure Deficit (VPD) Approximation
    es = 0.6108 * np.exp((17.27 * df['Temparature']) / (df['Temparature'] + 237.3))
    ea = (df['Humidity'] / 100) * es
    df['vpd_approx'] = es - ea

    # Temperature-Moisture Stress Index
    df['temp_moisture_stress'] = (df['Temparature'] - df['Temparature'].mean()) * (
                df['Moisture'].mean() - df['Moisture'])

    # --- Crop-Specific and Soil-Specific Interactions ---
    for nutrient in ['Nitrogen', 'Potassium', 'Phosphorous']:
        avg_nutrient_by_crop = df.groupby('Crop Type')[nutrient].transform('mean')
        df[f'{nutrient}_deviation_from_crop_avg'] = df[nutrient] - avg_nutrient_by_crop

    avg_moisture_by_soil = df.groupby('Soil Type')['Moisture'].transform('mean')
    df['moisture_deviation_from_soil_avg'] = df['Moisture'] - avg_moisture_by_soil

    # --- Binned Features ---
    # Binning continuous data to capture non-linear effects
    df['Temp_Binned'] = pd.cut(df['Temparature'], bins=4, labels=['Cool', 'Mild', 'Warm', 'Hot'])
    df['Humidity_Binned'] = pd.cut(df['Humidity'], bins=4,
                                   labels=['Low_Hum', 'Medium_Hum', 'High_Hum', 'Very_High_Hum'])
    df['Moisture_Binned'] = pd.cut(df['Moisture'], bins=4, labels=['Dry', 'Slightly_Dry', 'Moist', 'Wet'])
    df['Nitrogen_Binned'] = pd.cut(df['Nitrogen'], bins=4, labels=['Low_N', 'Medium_N', 'High_N', 'Very_High_N'])
    df['Potassium_Binned'] = pd.cut(df['Potassium'], bins=4, labels=['Low_K', 'Medium_K', 'High_K', 'Very_High_K'])
    df['Phosphorous_Binned'] = pd.cut(df['Phosphorous'], bins=4, labels=['Low_P', 'Medium_P', 'High_P', 'Very_High_P'])

    return df


def select_features(x_df, y_series, k=45, method='model'):
    """
    Selects the K best features using one of several methods.

    Args:
        x_df (pd.DataFrame): The input features.
        y_series (pd.Series): The target variable.
        k (int): The number of top features to select.
        method (str): The method to use: 'kbest', 'rfe', or 'model'.
    """
    print(f"Original number of features: {x_df.shape[1]}")
    print(f"Selecting the top {k} features using method: '{method}'...")

    if k > x_df.shape[1]:
        print(f"Warning: k={k} is greater than the number of features ({x_df.shape[1]}). Using all features.")
        k = x_df.shape[1]

    if method == 'kbest':
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(x_df, y_series)
        selected_mask = selector.get_support()

    elif method == 'rfe':
        # RFE is a wrapper method, it needs an estimator to work with.
        # A simple RandomForest is a good choice.
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        selector = selector.fit(x_df, y_series)
        X_new = x_df.loc[:, selector.support_]
        selected_mask = selector.support_

    elif method == 'model':
        # SelectFromModel uses feature importance's from a model.
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        estimator.fit(x_df, y_series)
        # We set the threshold to select the top k features based on importance.
        # Using a "median" or a specific float value is common. To get exactly k,
        # we can sort importance's and pick the k-th as the threshold.
        selector = SelectFromModel(estimator, prefit=True, max_features=k, threshold=-np.inf)
        X_new = selector.transform(x_df)
        selected_mask = selector.get_support()

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    selected_columns = x_df.columns[selected_mask]
    X_selected_df = pd.DataFrame(X_new, columns=selected_columns)
    print(f"Number of features after selection: {X_selected_df.shape[1]}")

    return X_selected_df

