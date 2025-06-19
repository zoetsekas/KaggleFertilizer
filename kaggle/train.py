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
from ray.tune import RunConfig, report, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from kaggle import FertilizerClassifier, add_features, select_features


# --- 1. Refactored Trainable Function with Internal K-Fold ---
def train_with_internal_kfold(config):
    """
    This function performs a full K-Fold cross-validation for a single
    hyperparameter configuration. It reports metrics after each fold to
    work with schedulers like ASHAScheduler.
    """
    # --- Data and K-Fold Setup ---
    X_kfold = ray.get(config["X_ref"])
    y_kfold = ray.get(config["y_ref"])

    n_splits = config["n_splits"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    # --- Loop over each fold ---
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_kfold, y_kfold)):
        print(f"--- Starting Fold {fold_idx + 1}/{n_splits} ---")

        X_train_fold, X_val_fold = X_kfold.iloc[train_index].values, X_kfold.iloc[val_index].values
        y_train_fold, y_val_fold = y_kfold.iloc[train_index].values, y_kfold.iloc[val_index].values

        train_dataset_torch = TensorDataset(torch.from_numpy(X_train_fold).float(),
                                            torch.from_numpy(y_train_fold).long().squeeze())
        val_dataset_torch = TensorDataset(torch.from_numpy(X_val_fold).float(),
                                          torch.from_numpy(y_val_fold).long().squeeze())

        train_loader = DataLoader(train_dataset_torch, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset_torch, batch_size=config["batch_size"])

        # --- Re-initialize Model and Optimizer for each fold ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FertilizerClassifier(
            input_size=config["input_size"],
            num_classes=config["num_classes"],
            l1_units=config["l1_units"],
            l2_units=config["l2_units"],
            l3_units=config["l3_units"],
            l4_units=config["l4_units"],
            dropout_rate=config["dropout_rate"],
            activation_fn_name=config["activation_fn"]
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        loss_fn = nn.CrossEntropyLoss()

        # --- Training Loop for the current fold ---
        for epoch in range(config["num_epochs"]):
            model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

        # --- Validation for the current fold ---
        model.eval()
        total_correct, total_samples, total_val_loss, total_ap_at_3 = 0, 0, 0, 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss_batch = loss_fn(outputs, labels)
                _, predicted_top1 = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted_top1 == labels).sum().item()
                total_val_loss += val_loss_batch.item()
                all_preds.extend(predicted_top1.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                _, top3_indices = torch.topk(outputs, 3, dim=1)
                for i in range(labels.size(0)):
                    if labels[i] in top3_indices[i]:
                        rank = (top3_indices[i] == labels[i]).nonzero(as_tuple=True)[0].item() + 1
                        total_ap_at_3 += 1.0 / rank

        # Store metrics for this fold
        fold_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        fold_avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        fold_map_at_3 = total_ap_at_3 / total_samples if total_samples > 0 else 0.0
        fold_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        fold_metrics.append({
            "val_loss": fold_avg_val_loss,
            "accuracy": fold_accuracy,
            "map_at_3": fold_map_at_3,
            "f1_score": fold_f1
        })
        print(
            f"--- Fold {fold_idx + 1} Metrics: Val Loss: {fold_avg_val_loss:.4f}, Accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f} ---")

        # --- Report CUMULATIVE average metrics to Ray Tune after each fold ---
        # This makes the trial compatible with early-stopping schedulers like ASHAScheduler
        avg_metrics = {
            "val_loss": np.mean([m["val_loss"] for m in fold_metrics]),
            "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "map_at_3": np.mean([m["map_at_3"] for m in fold_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in fold_metrics])
        }
        report(avg_metrics)


def load_best_model(config_path, model_path, device="cpu"):
    """
    Loads the best model from a saved state dictionary and configuration file.

    Args:
        config_path (str): Path to the best_tune_result.json file.
        model_path (str): Path to the best_model.pt file.
        device (str): The device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded model, ready for inference.
    """
    print(f"Loading model configuration from: {config_path}")
    with open(config_path, 'r') as f:
        saved_result = json.load(f)

    best_config = saved_result['best_config']

    print("Instantiating model with the best hyperparameters...")
    model = FertilizerClassifier(
        input_size=best_config["input_size"],
        num_classes=best_config["num_classes"],
        l1_units=best_config["l1_units"],
        l2_units=best_config["l2_units"],
        l3_units=best_config["l3_units"],
        l4_units=best_config["l4_units"],
        dropout_rate=best_config["dropout_rate"],
        activation_fn_name=best_config["activation_fn"]
    )

    print(f"Loading model state from: {model_path}")
    model_state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully and set to evaluation mode.")
    return model


def create_submission_file(test_data_path, submission_path, model_path, config_path, preprocessor_path,
                           label_encoder_path, selected_columns_path):
    """
    Loads test data, preprocesses it, makes predictions with the best model,
    and creates a submission.csv file.
    """
    print("\n--- Creating Submission File ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load all necessary artifacts ---
    model = load_best_model(config_path, model_path, device)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)
    with open(selected_columns_path, 'r') as f:
        selected_columns = json.load(f)

    # --- Load and Preprocess Test Data ---
    print(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    # Keep the 'id' column for the final submission file
    test_ids = test_df['id']

    print("Applying feature engineering to test data...")
    test_df = add_features(test_df)

    # Ensure test set has all columns needed for preprocessor, add missing ones with 0
    training_cols = preprocessor.feature_names_in_
    for col in training_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[training_cols]  # Ensure order is the same

    print("Applying preprocessing to test data...")
    X_test_processed_np = preprocessor.transform(test_df)

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=feature_names)

    # Filter to only the columns the model was trained on
    X_test_selected = X_test_processed[selected_columns]

    # --- Make Predictions ---
    print("Making predictions on the test set...")
    X_test_tensor = torch.from_numpy(X_test_selected.values).float().to(device)

    all_predictions = []
    with torch.no_grad():
        # Process in batches to avoid memory issues with large test sets
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=512)
        for batch in test_loader:
            features = batch[0]
            outputs = model(features)
            _, predicted_indices = torch.max(outputs, 1)
            all_predictions.extend(predicted_indices.cpu().numpy())

    # Inverse transform the predicted indices to get the original fertilizer names
    predicted_fertilizer_names = label_encoder.inverse_transform(all_predictions)

    # --- Create and Save Submission File ---
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Fertilizer Name': predicted_fertilizer_names
    })

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created successfully at: {submission_path}")

def pipeline(include_data_prep:bool=True, artifacts_dir: str = "run_artifacts"):
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(include_dashboard=False)

    # Define paths for saving artifacts
    output_dir = artifacts_dir
    os.makedirs(output_dir, exist_ok=True)
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    label_encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    selected_columns_path = os.path.join(output_dir, "selected_columns.json")

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
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y_labels), name=TARGET_COLUMN)
        NUM_CLASSES = len(label_encoder.classes_)
        joblib.dump(label_encoder, label_encoder_path)  # Save Label Encoder

        # --- Feature Preprocessing (One-Hot Encoding for categorical) ---
        # --- Feature Preprocessing with Scaling for numerical ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(feature_range=(-1, 1)), NUMERICAL_COLS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
            ], remainder='passthrough')

        preprocessor.fit(X_features)  # Fit preprocessor
        joblib.dump(preprocessor, preprocessor_path)  # Save Preprocessor
        print(f"Preprocessor saved to {preprocessor_path}")

        X_processed_np = preprocessor.transform(X_features)
        feature_names = preprocessor.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed_np, columns=feature_names)

        # Now you can choose the selection method here
        X_selected = select_features(X_processed, y_encoded, k=40, method='model')  # Changed to 'model'
        INPUT_SIZE = X_selected.shape[1]

        # Save the list of selected columns
        with open(selected_columns_path, 'w') as f:
            json.dump(X_selected.columns.tolist(), f)
        print(f"Selected column names saved to {selected_columns_path}")

        # Save the selected features and labels to a CSV file
        print(f"Saving {X_selected.shape[1]} selected features and labels to CSV...")
        final_df_to_save = pd.concat([X_selected, y_encoded], axis=1)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"selected_features_{timestamp}.csv"
        final_df_to_save.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")



    else:
        # This block now loads the most recently saved preprocessed file
        try:
            list_of_files = glob.glob(os.path.join(output_dir, 'selected_features_*.csv'))
            if not list_of_files:
                raise FileNotFoundError("No preprocessed file found. Please run with include_data_prep=True first.")
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"Loading data from most recent file: {latest_file}")
            loaded_df = pd.read_csv(latest_file)
            TARGET_COLUMN_NAME = 'Fertilizer Name'
            if TARGET_COLUMN_NAME not in loaded_df.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN_NAME}' not found in {latest_file}")

            y_encoded = loaded_df[TARGET_COLUMN_NAME]
            X_selected = loaded_df.drop(columns=[TARGET_COLUMN_NAME])
            INPUT_SIZE = X_selected.shape[1]
            NUM_CLASSES = len(y_encoded.unique())

        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            print("Exiting pipeline.")
            return

    print(f"Final Input feature size after selection: {INPUT_SIZE}, Number of classes: {NUM_CLASSES}")
    print(f"Data shapes: X={X_selected.shape}, y={y_encoded.shape}")

    # Put data into Ray's object store to be accessed by all trials efficiently
    X_ref = ray.put(X_selected)
    y_ref = ray.put(y_encoded)

    # 5. Define the Search Space for Hyperparameters
    search_space = {
        "input_size": tune.grid_search([INPUT_SIZE]),  # Fixed based on your data
        "num_classes": tune.grid_search([NUM_CLASSES]),  # Fixed based on your data
        "lr": tune.loguniform(1e-4, 1e-2),  # Sample learning rate logarithmically between 0.0001 and 0.01
        "batch_size": tune.choice([256, 512, 1024]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "activation_fn": tune.choice(["leaky_relu", "elu", "gelu", "silu"]),
        "l1_units": tune.choice([128, 256, 512]),
        "l2_units": tune.choice([64, 128, 256]),
        "l3_units": tune.choice([32, 64, 128]),
        "l4_units": tune.choice([8, 16, 32]),
        "optimizer": tune.choice(["AdamW"]),
        "num_epochs": 30, # Num epochs PER FOLD
        "n_splits": 5,    # Number of folds to use inside the trainable
        "X_ref": X_ref,
        "y_ref": y_ref
    }

    # --- Scheduler Setup ---
    scheduler = ASHAScheduler(
        max_t=search_space["n_splits"],  # Max "time" is the total number of folds
        grace_period=2,  # A trial must run for at least 2 folds before being stopped
        reduction_factor=2
    )

    resources = {"cpu": 2}
    if torch.cuda.is_available():
        resources = {"cpu": 2, "gpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(train_with_internal_kfold, resources),
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=20,  # Total number of hyperparameter combinations to test
            scheduler=scheduler,  # Add the scheduler here
            metric="val_loss",
            mode="min"
        ),
        run_config=RunConfig(name="fertilizer_internal_kfold_exp_asha")
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = os.path.join(output_dir, f"best_model_{timestamp}.pt")
    best_result_filename = os.path.join(output_dir, f"best_tune_result_{timestamp}.json")

    # Save best config and metrics to JSON
    best_config_to_save = {k: v for k, v in best_result.config.items() if k not in ['X_ref', 'y_ref']}
    result_to_save = {
        'best_config': best_config_to_save,
        'metrics': best_result.metrics,
        'model_path': model_filename,
        'preprocessor_path': preprocessor_path,
        'label_encoder_path': label_encoder_path,
        'selected_columns_path': selected_columns_path
    }

    with open(best_result_filename, 'w') as f:
        json.dump(result_to_save, f, indent=4)
    print(f"Best trial results saved to {best_result_filename}")

    if best_result.checkpoint:
        best_checkpoint_dict = best_result.checkpoint.to_dict()
        model_state_dict = best_checkpoint_dict["model_state_dict"]
        torch.save(model_state_dict, model_filename)
        print(f"Saved best model state to {model_filename}")
    else:
        print("No checkpoint found for the best trial. Could not save model.")

    print("\n" + "=" * 50)
    print("Ray Tune with Internal K-Fold Finished!")
    print("=" * 50)
    print(f"Best trial's final average validation loss: {best_result.metrics['val_loss']:.4f}")
    print(f"Best trial's final average accuracy: {best_result.metrics['accuracy']:.4f}")
    print(f"Best trial's final average F1 Score: {best_result.metrics['f1_score']:.4f}")
    print("\nBest hyperparameters found:")
    print(best_config_to_save)
    ray.shutdown()

    try:
        list_of_configs = glob.glob(os.path.join(output_dir, 'best_tune_result_*.json'))
        latest_config_file = max(list_of_configs, key=os.path.getctime)
        with open(latest_config_file, 'r') as f:
            saved_data = json.load(f)
            latest_model_file = saved_data['model_path']
        if os.path.exists(latest_model_file) and os.path.exists('../data/train.csv'):
            create_submission_file(
                test_data_path='../data/train.csv',
                submission_path="submission.csv",
                model_path=latest_model_file,
                config_path=latest_config_file,
                preprocessor_path=saved_data['preprocessor_path'],
                label_encoder_path=saved_data['label_encoder_path'],
                selected_columns_path=saved_data['selected_columns_path']
            )
        else:
            print("\nCould not create submission file: model or test data not found.")
    except (ValueError, FileNotFoundError) as e:
        print(f"\nCould not load a saved model or create submission: {e}")

if __name__ == '__main__':
    pipeline(include_data_prep=True)