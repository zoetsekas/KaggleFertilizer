import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from torch import nn


class FertilizerClassifier(nn.Module):
    def __init__(self, input_size, num_numerical_features, categorical_cardinalities, embedding_dim, num_hidden_layers, hidden_units,
                 num_classes, dropout_rate, activation_fn):
        super().__init__()

        self.activation_fn_name = activation_fn
        self.__get_activation_function__()
        self.input_size = input_size
        if self.input_size is None:
            self.num_numerical_features = num_numerical_features

            # Create a ModuleList to hold the embedding layers
            self.embedding_layers = nn.ModuleList(
                [nn.Embedding(num_categories, embedding_dim) for num_categories in categorical_cardinalities]
            )

            # Calculate the total size of the concatenated features (numerical + embedded categorical)
            total_embedding_dim = len(categorical_cardinalities) * embedding_dim
            combined_input_size = num_numerical_features + total_embedding_dim

            in_features = combined_input_size
        else:
            in_features = input_size

        layers = []
        # Dynamically create hidden layers
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units  # The input of the next layer is the output of this one

        # Add the final output layer
        layers.append(nn.Linear(in_features, num_classes))
        # Add a Softmax layer to get probability outputs
        layers.append(nn.Softmax(dim=1))

        # Create the sequential network from the list of layers
        self.network = nn.Sequential(*layers)

    def __get_activation_function__(self):
        # Select activation function based on string name
        if self.activation_fn_name == "relu":
            self.activation = nn.ReLU()
        elif self.activation_fn_name == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif self.activation_fn_name == "elu":
            self.activation = nn.ELU()
        elif self.activation_fn_name == "gelu":
            self.activation = nn.GELU()
        elif self.activation_fn_name == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn_name}")

    def forward(self, x):

        if self.input_size is None:
            # Split the input tensor into numerical and categorical parts
            x_numerical = x[:, :self.num_numerical_features]
            x_categorical = x[:, self.num_numerical_features:].long()

            # Get embeddings for each categorical feature
            embedded_cats = [
                self.embedding_layers[i](x_categorical[:, i]) for i in range(x_categorical.shape[1])
            ]

            # Concatenate numerical features with the embedded categorical features
            x_combined = torch.cat([x_numerical] + embedded_cats, dim=1)
            return self.network(x_combined)
        else:
            return self.network(x)


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
    # Feature 1: Crop Nutrient Requirement Scores
    crop_needs_map = {
        'N_Need_Score': {
            'Maize': 3, 'Sugarcane': 3, 'Cotton': 3, 'Wheat': 3, 'Barley': 3,
            'Paddy': 2, 'Tobacco': 2, 'Millets': 2, 'Oil seeds': 2,
            'Ground Nuts': 1, 'Pulses': 1
        },
        'P_Need_Score': {
            'Oil seeds': 3, 'Ground Nuts': 3, 'Pulses': 3,
            'Maize': 2, 'Cotton': 2, 'Wheat': 2, 'Barley': 2, 'Paddy': 2, 'Tobacco': 2,
            'Sugarcane': 1, 'Millets': 1
        },
        'K_Need_Score': {
            'Ground Nuts': 3, 'Pulses': 3,
            'Cotton': 2, 'Tobacco': 2, 'Oil seeds': 2,
            'Maize': 1, 'Sugarcane': 1, 'Wheat': 1, 'Barley': 1, 'Paddy': 1, 'Millets': 1
        }
    }
    df['N_Need_Score'] = df['Crop Type'].map(crop_needs_map['N_Need_Score'])
    df['P_Need_Score'] = df['Crop Type'].map(crop_needs_map['P_Need_Score'])
    df['K_Need_Score'] = df['Crop Type'].map(crop_needs_map['K_Need_Score'])

    # Feature 2: Nutrient Gap / Deficiency Features
    C = 20
    df['N_Gap'] = df['N_Need_Score'] * C - df['Nitrogen']
    df['P_Gap'] = df['P_Need_Score'] * C - df['Phosphorous']
    df['K_Gap'] = df['K_Need_Score'] * C - df['Potassium']

    # Feature 3: Soil Nutrient Ratio Features
    df['N_P_Ratio_Soil'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
    df['N_K_Ratio_Soil'] = df['Nitrogen'] / (df['Potassium'] + 1)
    df['P_K_Ratio_Soil'] = df['Phosphorous'] / (df['Potassium'] + 1)

    soil_retention_map = {'Clayey': 3, 'Black': 2, 'Loamy': 2, 'Red': 1, 'Sandy': 0}
    df['Moisture_Retention_Score'] = df['Soil Type'].map(soil_retention_map)
    df['Effective_Moisture'] = df['Moisture_Retention_Score'] * df['Moisture']
    df['Temp_Hum_Index'] = df['Temparature'] * df['Humidity']

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

    print("--- Adding Advanced Features for XGBoost ---")

    # Feature 5: Higher-Order Nutrient and Gap Features
    df['N_Gap_Squared'] = df['N_Gap'] ** 2
    df['P_Gap_Squared'] = df['P_Gap'] ** 2
    df['NP_Gap_Interaction'] = df['N_Gap'] * df['P_Gap']
    df['NK_Gap_Interaction'] = df['N_Gap'] * df['K_Gap']

    # Feature 6: Discretization (Binning) of Continuous Features
    df['Humidity_Bin'] = pd.cut(
        df['Humidity'],
        bins=[0, 60, 75, 100],
        labels=['Low_Humidity', 'Medium_Humidity', 'High_Humidity']
    )

    # --- 4. Feature Engineering from Reference Data Logic ---
    print("--- Adding Features based on Reference Data Logic ---")

    # Feature 7: Sulphur Need Score
    sulphur_needs_map = {
        'Oil seeds': 1, 'Ground Nuts': 1, 'Pulses': 1,
        'Maize': 0, 'Sugarcane': 0, 'Cotton': 0, 'Wheat': 0,
        'Barley': 0, 'Paddy': 0, 'Tobacco': 0, 'Millets': 0
    }
    df['Sulphur_Need_Score'] = df['Crop Type'].map(sulphur_needs_map).fillna(0)

    # Feature 8: Leaching Risk
    df['Leaching_Risk'] = 3 - df['Moisture_Retention_Score']

    # --- 5. Advanced Feature Engineering from Updated Reference ---
    print("--- Adding Features based on Updated Reference Data Logic ---")

    # Feature 9: Crop Seedling Sensitivity
    # Represents crop sensitivity to fertilizer burn (phytotoxicity) at the seedling stage.
    seedling_sensitivity_map = {
        'Maize': 2, 'Cotton': 2,
        'Sugarcane': 1, 'Wheat': 1, 'Barley': 1, 'Paddy': 1,
        'Millets': 1, 'Tobacco': 1, 'Pulses': 1, 'Ground Nuts': 1, 'Oil seeds': 1
    }
    df['Crop_Seedling_Sensitivity'] = df['Crop Type'].map(seedling_sensitivity_map).fillna(1)

    # Feature 10: Slow Release Benefit Score
    # Represents the combined benefit of a slow-release fertilizer based on leaching risk and crop duration.
    df['Slow_Release_Benefit_Score'] = df['Leaching_Risk'] + df['Crop Type'].apply(
        lambda x: 1 if x == 'Sugarcane' else 0)

    # df.drop(columns=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Total_NPK'], inplace=True)

    return df


def select_features(x_df, y_series, k=20, method='model'):
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

