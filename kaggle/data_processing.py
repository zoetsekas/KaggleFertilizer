import glob
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune import RunConfig, report, TuneConfig, Checkpoint, Tuner, CheckpointConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split

from kaggle import add_features

# https://www.kaggle.com/competitions/playground-series-s5e6/discussion/582515


def process_data(include_data_prep:bool=True, artifacts_dir: str = "run_artifacts", use_kfold: bool = True, use_embedding_layer=False):


    # Define paths for saving artifacts
    output_dir = artifacts_dir
    os.makedirs(output_dir, exist_ok=True)
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    label_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    selected_columns_path = os.path.join(output_dir, "selected_columns.json")

    num_numerical_features = None
    final_categorical_cardinalities = None
    X_final_for_tune = None
    y_encoded = None
    NUM_CLASSES = None

    if include_data_prep:
        df1 = pd.read_csv('../data/train.csv')
        df2 = pd.read_csv('../data/Fertilizer_Prediction.csv')
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop(columns=['id'])

        # Preprocessing
        df = add_features(df)

        # Drop the duplicate rows
        df = df.drop_duplicates()

        # Define the target column
        TARGET_COLUMN = 'Fertilizer Name'
        NUMERICAL_COLS = list(df.select_dtypes(include=np.number).columns)
        CATEGORICAL_COLS = list(df.select_dtypes(exclude=np.number).columns)
        if TARGET_COLUMN in CATEGORICAL_COLS: CATEGORICAL_COLS.remove(TARGET_COLUMN)
        # Separate features (X) and target (y)
        X_features = df.drop(columns=[TARGET_COLUMN])
        y_labels = df[TARGET_COLUMN]

        # --- Target Encoding ---
        # Encode the target variable 'Fertilizer Name' into numerical labels
        target_label_encoder = LabelEncoder()
        y_encoded = pd.Series(target_label_encoder.fit_transform(y_labels), name=TARGET_COLUMN)
        NUM_CLASSES = len(target_label_encoder.classes_)
        joblib.dump(target_label_encoder, label_encoder_path)
        print(f"Target Label Encoder saved to {label_encoder_path}")

        if not use_embedding_layer:
            # --- Feature Preprocessing (OrdinalEncoder for categorical) ---
            # --- Feature Preprocessing with Scaling for numerical ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(feature_range=(0, 1)), NUMERICAL_COLS),
                    ('one_hot', OneHotEncoder(handle_unknown='error'), CATEGORICAL_COLS)
                ], remainder='passthrough')

            X_processed_np = preprocessor.fit_transform(X_features)  # Fit preprocessor
            joblib.dump(preprocessor, preprocessor_path)  # Save Preprocessor
            print(f"Preprocessor saved to {preprocessor_path}")

            feature_names = list(preprocessor.get_feature_names_out())
            X_final_for_tune = pd.DataFrame(X_processed_np, columns=feature_names)
            print(
                f"Final Input: {feature_names} features")
        else:
            # --- Feature Preprocessing (OrdinalEncoder for categorical) ---
            # --- Feature Preprocessing with Scaling for numerical ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(feature_range=(-1, 1)), NUMERICAL_COLS),
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_COLS)
                ], remainder='passthrough')

            X_processed_np = preprocessor.fit_transform(X_features)  # Fit preprocessor
            joblib.dump(preprocessor, preprocessor_path)  # Save Preprocessor
            print(f"Preprocessor saved to {preprocessor_path}")

            # Get cardinalities for embedding layers
            categorical_cardinalities = [len(cat) for cat in preprocessor.named_transformers_['cat'].categories_]

            feature_names = list(preprocessor.get_feature_names_out())
            X_processed = pd.DataFrame(X_processed_np, columns=feature_names)

            X_selected = X_processed

            # X_selected = select_features(X_processed, y_encoded, k=40, method='model')
            #
            # with open(selected_columns_path, 'w') as f:
            #     json.dump(X_selected.columns.tolist(), f)
            # print(f"Selected column names saved to {selected_columns_path}")

            # After selection, re-calculate the info needed for the model
            final_numerical_cols = [col for col in X_selected.columns if col.startswith('num__')]
            final_categorical_cols = [col for col in X_selected.columns if col.startswith('cat__')]

            num_numerical_features = len(final_numerical_cols)

            # We need to map the selected categorical columns back to their original cardinalities
            # First get the original names from the processed names
            original_cat_names_from_selected = [col.split('__')[-1] for col in final_categorical_cols]
            # Then find the index of these original names in the full list of categorical columns
            final_cat_indices = [CATEGORICAL_COLS.index(name) for name in original_cat_names_from_selected]
            # Use these indices to get the correct cardinalities
            final_categorical_cardinalities = [categorical_cardinalities[i] for i in final_cat_indices]

            # Re-combine numerical and categorical for the final dataset, ensuring correct order
            X_final_for_tune = X_selected[final_numerical_cols + final_categorical_cols]

            print(
                f"Final Input: {num_numerical_features} numerical features, {len(final_categorical_cardinalities)} categorical features.")

        print(f"Saving {X_final_for_tune.shape[1]} selected features and labels to CSV...")
        final_df_to_save = pd.concat([X_final_for_tune.reset_index(drop=True), y_encoded.reset_index(drop=True)],
                                     axis=1)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"selected_features_{timestamp}.csv")
        # final_df_to_save.to_csv(output_filename, index=False)
        # print(f"Data saved to {output_filename}")
    else:
        pass

    return X_final_for_tune, y_encoded