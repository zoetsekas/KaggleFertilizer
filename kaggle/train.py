import os
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from kaggle import FertilizerClassifier, add_features, feature_selection


def train_func_with_tune(config):
    # num_epochs is per fold
    num_epochs_per_fold = config["num_epochs"]  # Can also be a hyperparameter
    n_splits = config.get("n_splits", 5)  # Get K for K-fold from config, default to 5

    # Access the dataset shards
    # ray.train.get_dataset_shard() works within ray.train, which TorchTrainer uses.
    # TorchTrainer is launched by Tune when used with tune.Tuner().
    # train_data_shard = ray.train.get_dataset_shard("train")
    # test_data_shard = ray.train.get_dataset_shard("test")  # For validation/reporting

    full_dataset_shard = ray.train.get_dataset_shard("dataset")  # Renamed from "train"
    # Convert Ray Dataset shard to Pandas DataFrame to use StratifiedKFold
    # This might be memory-intensive for very large datasets.
    # Consider if your dataset fits in memory on the worker.
    df_for_kfold = full_dataset_shard.to_pandas()

    # Extract features (X) and labels (y) for StratifiedKFold
    # Assuming 'labels' column exists and other columns are features
    X_kfold = df_for_kfold[[col for col in df_for_kfold.columns if col.startswith('feature_')]].values
    y_kfold = df_for_kfold["labels"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.get("fold_seed", 42))

    fold_metrics_list = []

    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_kfold, y_kfold)):
        print(
            f"Worker {ray.train.get_context().get_local_rank()} - Trial {ray.train.get_context().get_trial_name()} - Fold {fold_idx + 1}/{n_splits}")

        X_train_fold, X_val_fold = X_kfold[train_index], X_kfold[val_index]
        y_train_fold, y_val_fold = y_kfold[train_index], y_kfold[val_index]

        # Convert fold data back to Ray Datasets for iter_torch_batches
        # This is a bit of back-and-forth but allows using existing iter_torch_batches logic.
        # Alternatively, you could adapt the training loop to directly use NumPy arrays/PyTorch Tensors.
        train_fold_df = pd.DataFrame(X_train_fold, columns=[f'feature_{i}' for i in range(X_train_fold.shape[1])])
        train_fold_df['labels'] = y_train_fold
        train_fold_ray_dataset = ray.data.from_pandas(train_fold_df)
        # No need to map_batches again if X_train_fold is already the correct feature array

        val_fold_df = pd.DataFrame(X_val_fold, columns=[f'feature_{i}' for i in range(X_val_fold.shape[1])])
        val_fold_df['labels'] = y_val_fold
        val_fold_ray_dataset = ray.data.from_pandas(val_fold_df)

        # Instantiate the model with hyperparameters from 'config'
        model = FertilizerClassifier(
            input_size=config["input_size"],
            num_classes=config["num_classes"],
            l1_units=config["l1_units"],
            l2_units=config["l2_units"],
            l3_units=config["l3_units"],
            dropout_rate=config["dropout_rate"],
            activation_fn_name=config["activation_fn"]
        )

        # Choose optimizer based on config
        if config["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config.get("sgd_momentum", 0.9),
                                        weight_decay=config["weight_decay"])
        elif config["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        elif config["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        loss_fn = nn.CrossEntropyLoss()
        str_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(str_device)
        model.to(device)
        avg_train_loss_epoch = 0
        for epoch in range(num_epochs_per_fold):
            # --- Training Loop for the current fold ---
            model.train()
            total_train_loss = 0
            num_train_batches = 0

            train_dataset_torch = torch.utils.data.TensorDataset(
                torch.from_numpy(X_train_fold).float(),
                torch.from_numpy(y_train_fold).long()
            )
            train_loader_fold = torch.utils.data.DataLoader(
                train_dataset_torch, batch_size=config["batch_size"], shuffle=True
            )

            for features_tensor, labels_tensor in train_loader_fold:
                features = features_tensor.to(device)
                labels = labels_tensor.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss_epoch = total_train_loss / num_train_batches if num_train_batches > 0 else 0
            # Optionally print per-epoch-per-fold loss
            print(f"  Epoch {epoch+1}, Fold Train Loss: {avg_train_loss_epoch:.4f}")

        # --- Validation Loop for the current fold ---
        model.eval()
        total_correct = 0
        total_samples = 0
        total_val_loss = 0
        num_val_batches = 0
        total_ap_at_3 = 0.0

        val_dataset_torch = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_fold).float(),
            torch.from_numpy(y_val_fold).long()
        )
        val_loader_fold = torch.utils.data.DataLoader(
            val_dataset_torch, batch_size=config["batch_size"]
        )

        with torch.no_grad():
            for features_tensor, labels_tensor in val_loader_fold:
                features = features_tensor.to(device)
                labels = labels_tensor.to(device)

                outputs = model(features)
                val_loss_batch = loss_fn(outputs, labels)

                _, predicted_top1 = torch.max(outputs, 1)  # Use outputs directly
                total_samples += labels.size(0)
                total_correct += (predicted_top1 == labels).sum().item()
                total_val_loss += val_loss_batch.item()
                num_val_batches += 1

                _, top3_indices = torch.topk(outputs, 3, dim=1)
                for i in range(labels.size(0)):
                    true_label = labels[i]
                    predicted_top3_for_sample = top3_indices[i]
                    ap_at_3_for_sample = 0.0
                    for rank_idx, pred_idx in enumerate(predicted_top3_for_sample):
                        rank = rank_idx + 1
                        if pred_idx == true_label:
                            ap_at_3_for_sample = 1.0 / rank
                            break
                    total_ap_at_3 += ap_at_3_for_sample

        fold_accuracy = total_correct / total_samples if total_samples > 0 else 0
        fold_avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        fold_map_at_3 = total_ap_at_3 / total_samples if total_samples > 0 else 0

        fold_metrics_list.append({
            "loss": avg_train_loss_epoch,  # Or avg train loss over all epochs for this fold
            "val_loss": fold_avg_val_loss,
            "accuracy": fold_accuracy,
            "map_at_3": fold_map_at_3
        })
        print(
            f"  Fold {fold_idx + 1} Metrics: Val Loss: {fold_avg_val_loss:.4f}, Accuracy: {fold_accuracy:.4f}, MAP@3: {fold_map_at_3:.4f}")

        # Aggregate metrics across folds
    if fold_metrics_list:
        avg_loss_all_folds = np.mean([m["loss"] for m in fold_metrics_list])
        avg_val_loss_all_folds = np.mean([m["val_loss"] for m in fold_metrics_list])
        avg_accuracy_all_folds = np.mean([m["accuracy"] for m in fold_metrics_list])
        avg_map_at_3_all_folds = np.mean([m["map_at_3"] for m in fold_metrics_list])
    else:  # Should not happen if n_splits > 0
        avg_loss_all_folds = 0
        avg_val_loss_all_folds = 0
        avg_accuracy_all_folds = 0
        avg_map_at_3_all_folds = 0

    print(
        f"Worker {ray.train.get_context().get_local_rank()} - Trial End. Avg Val Loss: {avg_val_loss_all_folds:.4f}, Avg Accuracy: {avg_accuracy_all_folds:.4f}, Avg MAP@3: {avg_map_at_3_all_folds:.4f}")

    ray.train.report(metrics={
        "loss": avg_loss_all_folds,  # This is now average training loss across folds
        "val_loss": avg_val_loss_all_folds,
        "accuracy": avg_accuracy_all_folds,
        "map_at_3": avg_map_at_3_all_folds
    })

if __name__ == '__main__':
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    df1 = pd.read_csv('../data/train.csv')
    df2 = pd.read_csv('../data/Fertilizer_Prediction.csv')
    df = pd.concat([df1, df2], ignore_index=True)

    df = df.drop(columns=['id'])

    df = add_features(df)

    # sample:pd.DataFrame = df.head(5)
    # now = datetime.now()
    # timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # sample.to_csv(f'../data/{timestamp}_sample.csv', index=False)
    #
    # df.to_csv(f'../data/{timestamp}_augmented.csv', index=False)
    #
    # print("\nDataFrame Descriptive Statistics (Numerical Columns):")
    # print(df.describe())
    # df.describe().to_csv(f'../data/{timestamp}_describe.csv', index=True)

    print("\nValue Counts for Categorical Columns:")
    print("Soil_Type:\n", df['Soil Type'].value_counts())
    print("\nCrop_Type:\n", df['Crop Type'].value_counts())
    print("\nFertilizer_Name:\n", df['Fertilizer Name'].value_counts())
    print("Temparature_Binned:\n", df['Temparature_Binned'].value_counts())
    print("\nHumidity_Binned:\n", df['Humidity_Binned'].value_counts())
    print("\nMoisture_Binned:\n", df['Moisture_Binned'].value_counts())
    print("\nNitrogen_Binned:\n", df['Nitrogen_Binned'].value_counts())
    print("\nPotassium_Binned:\n", df['Potassium_Binned'].value_counts())
    print("\nPhosphorous_Binned:\n", df['Phosphorous_Binned'].value_counts())

    # exit()

    # Preprocessing
    # Define the target column
    TARGET_COLUMN = 'Fertilizer Name'

    # Define numerical and categorical features (excluding 'id' and the target)
    NUMERICAL_COLS = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
                      'NP_Ratio', 'NK_Ratio', 'PK_Ratio', 'Total_NPK',
                      'N_Percentage', 'P_Percentage', 'K_Percentage',
                      'Temp_Moisture_Interaction', 'Temp_Humidity_Interaction', 'Hum_Moisture_Interaction',
                      'Temperature_Squared', 'Humidity_Squared', 'Moisture_Squared'
                      ]
    CATEGORICAL_COLS = ['Soil Type', 'Crop Type',
                        'Temparature_Binned', 'Humidity_Binned', 'Moisture_Binned',
                        'Potassium_Binned', 'Phosphorous_Binned', 'Nitrogen_Binned']

    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Select the best features
    X = feature_selection(input_df=X, target_df=y)

    # Drop the duplicate rows
    X = X.drop_duplicates()

    numerical_columns = []
    for current_column in NUMERICAL_COLS:
        if current_column in X.columns:
            numerical_columns.append(current_column)

    categorical_columns = []
    for current_column in CATEGORICAL_COLS:
        if current_column in X.columns:
            categorical_columns.append(current_column)


    # --- Target Encoding ---
    # Encode the target variable 'Fertilizer Name' into numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    NUM_CLASSES = len(label_encoder.classes_)  # Global variable for num_classes
    print(f"Detected {NUM_CLASSES} unique fertilizer names: {label_encoder.classes_}")

    # --- Feature Preprocessing (One-Hot Encoding for categorical) ---
    # --- Feature Preprocessing with Scaling for numerical ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(feature_range=(-1, 1)), numerical_columns),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='drop'  # Drop any columns not explicitly transformed (like 'id' if it wasn't dropped already)
    )

    # Fit and transform the features
    X_processed = preprocessor.fit_transform(X)

    # Determine the input size for the model based on the processed features
    INPUT_SIZE = X_processed.shape[1]  # Global variable for input_size
    print(f"Input feature size after preprocessing: {INPUT_SIZE}")

    # Combine processed features and encoded labels into a DataFrame for Ray Dataset
    # We convert the sparse matrix from OneHotEncoder to a dense array for easier handling.
    processed_data_for_ray = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed,
                                          columns=[f'feature_{i}' for i in range(INPUT_SIZE)])
    processed_data_for_ray['labels'] = y_encoded

    # Convert Pandas DataFrame to Ray Dataset
    # The `map_batches` step ensures the dataset yields dictionaries with 'features' (NumPy array)
    # and 'labels' (NumPy array) as expected by iter_torch_batches.
    ray_dataset = ray.data.from_pandas(processed_data_for_ray)

    # Transform the Ray Dataset to have 'features' (all input columns as a single array) and 'labels'
    ray_dataset = ray_dataset.map_batches(
        lambda batch: {
            "features": batch[[col for col in batch.columns if col.startswith('feature_')]].values.astype(np.float32),
            "labels": batch["labels"].values.astype(np.int64)  # Labels should be int64 for PyTorch LongTensor
        },
        batch_format="pandas"  # Process batches as Pandas DataFrames
    )

    # --- Split the Ray Dataset into Train and Test ---
    # Use .train_test_split() with a test_size of 0.2 (20%)
    # random_state ensures reproducibility of the split.
    # train_ray_dataset, test_ray_dataset = ray_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)



    # Split the Ray Dataset into 80% training and 20% testing sets
    train_dataset, test_dataset = ray_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    print(f"\nTotal Ray Dataset created with {ray_dataset.count()} rows.")
    print(f"Train Ray Dataset created with {train_dataset.count()} rows (80%).")
    print(f"Test Ray Dataset created with {test_dataset.count()} rows (20%).")

    # Example of peeking into the train dataset to verify structure and data types
    print("\n--- Sample from Training Dataset ---")
    for i, row in enumerate(train_dataset.take(1)):  # Take only 1 for brevity
        label_dtype = getattr(row['labels'], 'dtype', type(row['labels']).__name__)
        print(
            f"Sample {i + 1}: Features shape {row['features'].shape}, Features dtype: {row['features'].dtype}, Label: {row['labels']}, Label dtype: {label_dtype}")
        print(f"Sample {i + 1} Features (first 5 elements): {row['features'][:5]}")

    print("\n" + "=" * 50 + "\n")

    # 5. Define the Search Space for Hyperparameters
    search_space = {
        "input_size": tune.grid_search([INPUT_SIZE]),  # Fixed based on your data
        "num_classes": tune.grid_search([NUM_CLASSES]),  # Fixed based on your data
        "lr": tune.loguniform(1e-4, 1e-2),  # Sample learning rate logarithmically between 0.0001 and 0.01
        "batch_size": tune.choice([128, 256, 512]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "activation_fn": tune.choice(["relu", "leaky_relu", "elu", "gelu", "silu"]),
        "l1_units": tune.choice([64, 128, 256]),
        "l2_units": tune.choice([32, 64, 128]),
        "l3_units": tune.choice([8, 16, 32]),
        "optimizer": tune.choice(["Adam", "SGD", "AdamW"]),
        # "num_epochs": 30,  # Fixed number of epochs for each trial for simplicity
        "num_epochs": 15,  # Number of epochs PER FOLD
        "n_splits": 5,  # Number of folds for StratifiedKFold (can also be tuned: tune.choice([3, 5]))
        "fold_seed": tune.randint(0, 1000)  # Optional: vary seed for k-fold split per trial
    }

    USE_GPU = False
    if torch.cuda.is_available():
        print("GPU is available!!!!")
        USE_GPU = True

    # 6. Configure and Run Ray Tune
    # Use TorchTrainer within the tune.Tuner

    # For StratifiedKFold within each trial, num_workers=1 is often simplest
    # as the K-fold logic runs sequentially on that one worker.
    # If you use num_workers > 1, you need a more complex strategy for distributing folds.
    if USE_GPU:
        scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1})
    else:
        scaling_config = ScalingConfig(num_workers=1, use_gpu=False,
                                       resources_per_worker={"CPU": 4})  # Or more CPUs

    trainable = TorchTrainer(
        train_loop_per_worker=train_func_with_tune,
        scaling_config=scaling_config,
        # Pass the FULL dataset. It will be named "dataset" inside train_func_with_tune
        datasets={"dataset": ray_dataset}
    )

    scheduler = ASHAScheduler(
        metric="val_loss",  # This will be average val_loss across folds
        mode="min",
        max_t=search_space["num_epochs"] * search_space["n_splits"],  # Max "time" can be total epochs
        grace_period=search_space["n_splits"],  # Min "time" before stopping (e.g., complete one full CV)
        reduction_factor=2
    )

    tuner = tune.Tuner(
        trainable=trainable,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            num_samples=10,  # Number of different hyperparameter combinations to try
            scheduler=scheduler
        ),
        run_config=RunConfig(
            name="fertilizer_kfold_experiment",
            checkpoint_config=tune.CheckpointConfig(
                # Checkpointing with K-fold is more complex.
                # You might checkpoint the model after all K folds,
                # or the best model from one of the folds.
                # For now, let's assume we checkpoint based on the avg val_loss.
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
                num_to_keep=1,
            ),
        ),
    )

    # Run the tuning experiment
    results = tuner.fit()

    # Get the best trial
    best_result = results.get_best_result("val_loss", "min")

    print(f"\nBest trial found: {best_result.config['train_loop_config']}")
    print(f"Best trial final validation loss: {best_result.metrics['val_loss']:.4f}")
    print(f"Best trial final accuracy: {best_result.metrics['accuracy']:.4f}")
    print(f"Best trial final MAP@3: {best_result.metrics['map_at_3']:.4f}")

    # Model loading part needs adjustment if checkpointing is done within train_func_with_tune
    # Ray Train's TorchTrainer handles checkpointing automatically if configured in RunConfig.
    # The best_result.checkpoint will be a Ray Train Checkpoint object.
    if best_result.checkpoint:
        print(f"Best trial checkpoint path: {best_result.checkpoint.path}")
        # To load a model from a Ray Train checkpoint:
        # best_checkpoint = best_result.checkpoint
        # loaded_model_state = best_checkpoint.to_dict() # This loads the state dict
        # model_to_load = FertilizerClassifier(
        #     input_size=best_result.config['train_loop_config']["input_size"],
        #     num_classes=best_result.config['train_loop_config']["num_classes"],
        #     # ... other model params from best_result.config ...
        #     l1_units=best_result.config['train_loop_config']["l1_units"],
        #     l2_units=best_result.config['train_loop_config']["l2_units"],
        #     dropout_rate=best_result.config['train_loop_config']["dropout_rate"],
        #     activation_fn_name=best_result.config['train_loop_config']["activation_fn"]
        # )
        # model_to_load.load_state_dict(loaded_model_state['model_state_dict']) # Assuming 'model_state_dict' is the key
        # print("Model loaded from best checkpoint.")
        # Note: Ray's TorchTrainer saves checkpoints in a specific format.
        # You might need to adjust how you save/load if you implement manual checkpointing
        # inside `train_func_with_tune` with `ray.train.report(..., checkpoint=...)`.
        # The default TorchTrainer checkpoint will contain the model state dict.
        # For simplicity, the loading part is commented out as it depends on how
        # FertilizerClassifier and checkpointing are precisely set up.
        # The key is that `best_result.checkpoint` gives you access.
    else:
        print("No checkpoint found for the best trial.")

    ray.shutdown()

    print(f"\nBest trial found: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['val_loss']:.4f}")
    print(f"Best trial final accuracy: {best_result.metrics['accuracy']:.4f}")

    # You can also access the best model checkpoint if you configured checkpointing in your train_func_with_tune
    if best_result.path:
        print(f"Best trial checkpoint path: {best_result.path}")
        # You would load the model from this checkpoint for inference
        model_to_load = FertilizerClassifier(
            input_size=best_result.config['train_loop_config']["input_size"],
            num_classes=best_result.config['train_loop_config']["num_classes"],
            # ... other model params from best_result.config ...
            l1_units=best_result.config['train_loop_config']["l1_units"],
            l2_units=best_result.config['train_loop_config']["l2_units"],
            dropout_rate=best_result.config['train_loop_config']["dropout_rate"],
            activation_fn_name=best_result.config['train_loop_config']["activation_fn"]
        )
        model_to_load.load_state_dict(torch.load(os.path.join(best_result.path, "model.pt")))