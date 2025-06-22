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
from ray.tune import RunConfig, report, TuneConfig, Checkpoint
from ray.tune.schedulers import ASHAScheduler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
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
    best_fold_map_at_3 = -1.0  # Initialize to a low value for maximization
    best_model_state_for_trial = None
    avg_metrics = {}

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
            num_numerical_features=config["num_numerical_features"],
            categorical_cardinalities=config["categorical_cardinalities"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_hidden_layers=config["num_hidden_layers"],
            hidden_units=config["hidden_units"],
            dropout_rate=config["dropout_rate"],
            activation_fn_name=config["activation_fn"]
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        if config["loss_function"] == "CrossEntropyLoss":
            loss_fn = nn.CrossEntropyLoss()
        elif config["loss_function"] == "CosineEmbeddingLoss":
            loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss function: {config['loss_function']}")

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
            f"--- Fold {fold_idx + 1} Metrics: Val Loss: {fold_avg_val_loss:.4f}, Accuracy: {fold_accuracy:.4f}, MAP@3: {fold_map_at_3:.4f} ---")

        # Check if this fold's model is the best one seen so far in this trial based on MAP@3
        if fold_map_at_3 > best_fold_map_at_3:
            best_fold_map_at_3 = fold_map_at_3
            best_model_state_for_trial = model.state_dict()

        # --- Report CUMULATIVE average metrics to Ray Tune after each fold ---
        avg_metrics = {
            "val_loss": np.mean([m["val_loss"] for m in fold_metrics]),
            "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "map_at_3": np.mean([m["map_at_3"] for m in fold_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in fold_metrics])
        }
        report(avg_metrics)

    # --- After all folds, create a final checkpoint from the best model state ---
    if best_model_state_for_trial:
        checkpoint = Checkpoint.from_dict({"model_state_dict": best_model_state_for_trial})
        # The last reported metrics will be the final ones for the trial.
        # We report them again here with the checkpoint.
        report(metrics=avg_metrics, checkpoint=checkpoint)


def load_best_model(config_path, model_path, device="cpu"):
    """
    Loads the best model from a saved state dictionary and configuration file.
    """
    print(f"Loading model configuration from: {config_path}")
    with open(config_path, 'r') as f:
        saved_result = json.load(f)

    best_config = saved_result['best_config']

    print("Instantiating model with the best hyperparameters...")
    model = FertilizerClassifier(
        num_numerical_features=best_config["num_numerical_features"],
        categorical_cardinalities=best_config["categorical_cardinalities"],
        embedding_dim=best_config["embedding_dim"],
        num_hidden_layers=best_config["num_hidden_layers"],
        hidden_units=best_config["hidden_units"],
        dropout_rate=best_config["dropout_rate"],
        activation_fn_name=best_config["activation_fn"],
        num_classes=best_config["num_classes"]
    )

    print(f"Loading model state from: {model_path}")
    model_state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()
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

    model = load_best_model(config_path, model_path, device)
    preprocessor = joblib.load(preprocessor_path)
    target_label_encoder = joblib.load(label_encoder_path)
    with open(selected_columns_path, 'r') as f:
        selected_columns = json.load(f)

    print(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    test_ids = test_df['id']

    print("Applying feature engineering to test data...")
    test_df_features = add_features(test_df.drop(columns=['id']))

    print("Applying preprocessing to test data...")
    X_test_processed_np = preprocessor.transform(test_df_features)
    feature_names = preprocessor.get_feature_names_out()
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=feature_names)
    X_test_selected = X_test_processed[selected_columns]

    print("Making predictions on the test set...")
    X_test_tensor = torch.from_numpy(X_test_selected.values).float().to(device)

    all_predictions = []
    with torch.no_grad():
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=512)
        for batch in test_loader:
            features = batch[0]
            outputs = model(features)
            _, predicted_indices = torch.max(outputs, 1)
            all_predictions.extend(predicted_indices.cpu().numpy())

    predicted_fertilizer_names = target_label_encoder.inverse_transform(all_predictions)

    submission_df = pd.DataFrame({'id': test_ids, 'Fertilizer Name': predicted_fertilizer_names})
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

        # --- Feature Preprocessing (OrdinalEncoder for categorical) ---
        # --- Feature Preprocessing with Scaling for numerical ---
        # Using OrdinalEncoder instead of OneHotEncoder
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

        print(f"Saving {X_final_for_tune.shape[1]} selected features and labels to CSV...")
        final_df_to_save = pd.concat([X_final_for_tune.reset_index(drop=True), y_encoded.reset_index(drop=True)],
                                     axis=1)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = os.path.join(output_dir, f"selected_features_{timestamp}.csv")
        final_df_to_save.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
    else:
        pass

    print(
        f"Final Input: {num_numerical_features} numerical features, {len(final_categorical_cardinalities)} categorical features.")

    X_ref = ray.put(X_final_for_tune)
    y_ref = ray.put(y_encoded)

    # 5. Define the Search Space for Hyperparameters
    search_space = {
        "num_numerical_features": num_numerical_features,
        "categorical_cardinalities": final_categorical_cardinalities,
        "embedding_dim": tune.choice([8, 16]),
        "num_classes": NUM_CLASSES,
        "lr": tune.loguniform(1e-4, 1e-2),  # Sample learning rate logarithmically between 0.0001 and 0.01
        "batch_size": tune.choice([256, 512, 1024]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "activation_fn": tune.choice(["leaky_relu", "gelu", "silu"]),
        "num_hidden_layers": tune.choice([4, 6, 8]),
        "hidden_units": tune.choice([64, 128, 256]),
        "optimizer": tune.choice(["AdamW"]),
        "loss_function": tune.choice(["CrossEntropyLoss"]),
        "num_epochs": 30, # Num epochs PER FOLD
        "n_splits": 10,    # Number of folds to use inside the trainable
        "X_ref": X_ref,
        "y_ref": y_ref
    }

    # --- Scheduler Setup ---
    scheduler = ASHAScheduler(
        max_t=search_space["n_splits"],  # Max "time" is the total number of folds
        grace_period=2,  # A trial must run for at least 2 folds before being stopped
        reduction_factor=2
    )

    resources = {"cpu": 8}
    if torch.cuda.is_available():
        resources = {"cpu": 8, "gpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(train_with_internal_kfold, resources),
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=5,  # Total number of hyperparameter combinations to test
            scheduler=scheduler,  # Add the scheduler here
            metric="map_at_3",
            mode="max"
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
        if os.path.exists(latest_model_file) and os.path.exists('../data/test.csv'):
            create_submission_file(
                test_data_path='../data/test.csv',
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