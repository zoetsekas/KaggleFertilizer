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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from kaggle import FertilizerClassifier, add_features


def train_func_with_tune(config):
    num_epochs = config["num_epochs"]  # Can also be a hyperparameter

    # Access the dataset shards
    # ray.train.get_dataset_shard() works within ray.train, which TorchTrainer uses.
    # TorchTrainer is launched by Tune when used with tune.Tuner().
    train_data_shard = ray.train.get_dataset_shard("train")
    test_data_shard = ray.train.get_dataset_shard("test")  # For validation/reporting

    # Instantiate the model with hyperparameters from 'config'
    model = FertilizerClassifier(
        input_size=config["input_size"],
        num_classes=config["num_classes"],
        l1_units=config["l1_units"],
        l2_units=config["l2_units"],
        dropout_rate=config["dropout_rate"],
        activation_fn_name=config["activation_fn"]
    )

    # Choose optimizer based on config
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config.get("sgd_momentum", 0.9),
                                    weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    loss_fn = nn.CrossEntropyLoss()

    str_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str_device)
    model.to(device)



    for epoch in range(num_epochs):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        for batch in train_data_shard.iter_torch_batches(batch_size=config["batch_size"], dtypes=torch.float32,
                                                         device=str_device):
            features = batch["features"]
            labels = batch["labels"].long()

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches

        # --- Validation Loop (using test_data_shard) ---
        model.eval()  # Set model to evaluation mode
        total_correct = 0
        total_samples = 0
        total_val_loss = 0
        num_val_batches = 0
        total_ap_at_3 = 0.0  # Initialize sum of AP@3 scores

        with torch.no_grad():
            for batch in test_data_shard.iter_torch_batches(batch_size=config["batch_size"], dtypes=torch.float32,
                                                            device=str_device):
                features = batch["features"]
                labels = batch["labels"].long()

                outputs = model(features)
                val_loss = loss_fn(outputs, labels)

                # Accuracy calculation
                _, predicted_top1 = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted_top1 == labels).sum().item()
                total_val_loss += val_loss.item()
                num_val_batches += 1

                # MAP@3 Calculation
                # Get top-3 predicted class indices.
                # `outputs` are logits, shape (batch_size, num_classes)
                _, top3_indices = torch.topk(outputs, 3, dim=1) # top3_indices shape: (batch_size, 3)

                for i in range(labels.size(0)): # Iterate over samples in the batch
                    true_label = labels[i]
                    predicted_top3_for_sample = top3_indices[i] # Tensor of 3 predicted indices for this sample

                    ap_at_3_for_sample = 0.0
                    for rank_idx, pred_idx in enumerate(predicted_top3_for_sample):
                        rank = rank_idx + 1 # Convert 0-indexed to 1-indexed rank
                        if pred_idx == true_label:
                            ap_at_3_for_sample = 1.0 / rank
                            break # True label found, AP@3 is set for this sample
                    total_ap_at_3 += ap_at_3_for_sample

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        map_at_3 = total_ap_at_3 / total_samples if total_samples > 0 else 0

        print(f"Worker {ray.train.get_context().get_local_rank()} - Epoch {epoch + 1}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, MAP@3: {map_at_3:.4f}")

        # Report metrics to Ray Tune
        ray.train.report(metrics={
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": accuracy,
            "map_at_3": map_at_3  # Add MAP@3 to reported metrics
        })

if __name__ == '__main__':
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    df = pd.read_csv('../data/train.csv')

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
    CATEGORICAL_COLS = ['Soil Type', 'Crop Type']

    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

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
            ('num', MinMaxScaler(feature_range=(-1, 1)), NUMERICAL_COLS),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
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
        "activation_fn": tune.choice(["relu", "leaky_relu", "elu"]),
        "l1_units": tune.choice([64, 128, 256]),
        "l2_units": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["Adam", "SGD"]),
        "num_epochs": 30  # Fixed number of epochs for each trial for simplicity
    }

    USE_GPU = False
    if torch.cuda.is_available():
        print("GPU is available!!!!")
        USE_GPU = True

    # 6. Configure and Run Ray Tune
    # Use TorchTrainer within the tune.Tuner

    if USE_GPU:
        trainable = TorchTrainer(
            train_loop_per_worker=train_func_with_tune,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=USE_GPU, resources_per_worker={"CPU": 4, "GPU": 1}),
            datasets={"train": train_dataset, "test": test_dataset}  # Pass both datasets
        )
    else:
        trainable = TorchTrainer(
            train_loop_per_worker=train_func_with_tune,
            scaling_config=ScalingConfig(num_workers=4, use_gpu=USE_GPU),
            datasets={"train": train_dataset, "test": test_dataset}  # Pass both datasets
        )

    # Optional: Define a scheduler for early stopping (e.g., ASHA)
    # This helps stop bad performing trials early to save resources.
    scheduler = ASHAScheduler(
        metric="val_loss",  # Metric to monitor for early stopping
        mode="min",  # We want to minimize validation loss
        max_t=30,  # Max epochs per trial (should match num_epochs in config)
        grace_period=1,  # Don't stop trials before 1 epoch
        reduction_factor=2  # Factor for reducing active trials
    )

    tuner = tune.Tuner(
        trainable,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            num_samples=6,  # Number of different hyperparameter combinations to try
            scheduler=scheduler  # Apply the scheduler
        ),
        run_config=RunConfig(
            name="torch_trainer_experiment",
            # storage_path=output_dir,  # Set the desired output directory
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="loss",  # Metric to use for best checkpoint
                checkpoint_score_order="min",  # Smaller loss is better
                num_to_keep=1,  # Only keep the single best checkpoint
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