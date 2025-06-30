import logging
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from ray import tune, ray
from ray.tune import RunConfig, TuneConfig, Tuner
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

# --- Configuration & Constants ---
# It's good practice to define constants at the top of the script.
TARGET_COLUMN = 'Fertilizer Name'
N_SPLITS = 5  # Number of folds for cross-validation
NUM_BOOST_ROUND = 100_000  # Max boosting rounds
EARLY_STOPPING_ROUNDS = 50  # Early stopping rounds
TUNE_NUM_SAMPLES = 25  # Number of hyperparameter samples to try in Tune

# Configure logging for cleaner output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Custom Metric Calculation ---
# This section is well-written. I've added more comments for clarity.
def _calculate_map_at_3(y_true_labels: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Core logic for calculating Mean Average Precision at 3 (MAP@3)."""
    # Ensure y_true is integer type for comparison
    y_true_labels = y_true_labels.astype(int)

    # Reshape prediction array if it's flattened (as it is during xgb.train)
    if y_pred_probs.ndim == 1:
        num_samples = len(y_true_labels)
        if num_samples == 0:
            return 0.0
        # Infer number of classes from the flattened array length
        num_classes = len(y_pred_probs) // num_samples
        y_pred_probs = y_pred_probs.reshape(num_samples, num_classes)

    # Get the indices of the top 3 predictions by sorting probabilities in ascending order
    # and taking the last 3 columns.
    top3_preds_indices = np.argsort(y_pred_probs, axis=1)[:, -3:]

    total_ap_at_3 = 0.0
    for i in range(len(y_true_labels)):
        true_label = y_true_labels[i]
        # Predictions are sorted ascending, so reverse to get descending order (highest prob first)
        predicted_top3_for_sample = top3_preds_indices[i][::-1]

        ap_at_3_for_sample = 0.0
        if true_label in predicted_top3_for_sample:
            # Find the 1-based rank of the true label in the top 3 predictions
            rank = np.where(predicted_top3_for_sample == true_label)[0][0] + 1
            ap_at_3_for_sample = 1.0 / rank
        total_ap_at_3 += ap_at_3_for_sample

    return total_ap_at_3 / len(y_true_labels) if len(y_true_labels) > 0 else 0.0


def map_at_3(y_pred: np.ndarray, y_true: xgb.DMatrix) -> list[tuple[str, float]]:
    """XGBoost `feval` compatible wrapper for MAP@3 calculation."""
    true_labels = y_true.get_label()
    map_score = _calculate_map_at_3(true_labels, y_pred)
    # FIX: Use an underscore instead of the @ symbol
    return [('map_at_3', map_score)]


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(train_path: str, fertilizer_path: str) -> tuple[
    pd.DataFrame, pd.Series, LabelEncoder, ColumnTransformer]:
    """Loads, merges, and preprocesses the data."""
    logging.info("Loading and preprocessing data...")
    df1 = pd.read_csv(train_path)
    df2 = pd.read_csv(fertilizer_path)
    df = pd.concat([df1, df2], ignore_index=True)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Placeholder for your custom feature engineering
    # from kaggle import add_features
    # df = add_features(df)

    y_labels = df[TARGET_COLUMN]
    x = df.drop(columns=[TARGET_COLUMN])

    # Target Encoding
    target_encoder = LabelEncoder()
    y_encoded = pd.Series(target_encoder.fit_transform(y_labels), name=TARGET_COLUMN)

    # Feature Preprocessing
    categorical_cols = list(x.select_dtypes(include=['object', 'category']).columns)

    # IMPROVEMENT: Use handle_unknown='ignore' for robustness against unseen categories
    # in validation/test sets.
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")  # Makes output a DataFrame

    x_processed = preprocessor.fit_transform(x)

    logging.info(f"Data preprocessed. Shape of X: {x_processed.shape}")
    return x_processed, y_encoded, target_encoder, preprocessor


# --- Ray Tune Training Function ---
# In D:/projects/KaggleFertilizer/kaggle/xgboost_train.py

def train_xgboost(config: dict, x_data: pd.DataFrame, y_data: pd.Series):
    """
    Trainable function for Ray Tune.
    - Fetches data from Ray's object store.
    - Runs Stratified K-Fold CV.
    - Reports average metrics back to Tune.
    """
    if isinstance(y_data, (pd.DataFrame, pd.Series)):
        y_data = y_data.values.ravel()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    map3_scores, logloss_scores = [], []
    num_class = len(np.unique(y_data))

    # --- IMPROVEMENT: Dynamically build the params dictionary ---
    # Start with parameters common to all boosters
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': num_class,
        'booster': config.get('booster'),
        # --- End of new parameters ---
        'eta': config.get('learning_rate'),
        'max_depth': config.get('max_depth'),
        'subsample': config.get('subsample'),
        'colsample_bytree': config.get('colsample_bytree'),
        'lambda': config.get('reg_lambda'),
        'alpha': config.get('reg_alpha'),
        'gamma': config.get('gamma'),
        'min_child_weight': config.get('min_child_weight'),
        'colsample_bylevel': config.get('colsample_bylevel'),
        'colsample_bynode': config.get('colsample_bynode'),
        'seed': 42,
        'device': 'cuda',
        'tree_method': 'hist'
    }

    # Add DART-specific parameters only if the booster is 'dart'
    if params['booster'] == 'dart':
        params['rate_drop'] = config.get('rate_drop')
        params['skip_drop'] = config.get('skip_drop')

    for fold, (train_index, val_index) in enumerate(skf.split(x_data, y_data)):
        X_train, X_val = x_data.iloc[train_index], x_data.iloc[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=x_data.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=x_data.columns.tolist())

        evals_result = {}
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dval, 'validation')],
            evals_result=evals_result,
            custom_metric=map_at_3,
            verbose_eval=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            callbacks=[
                TuneReportCheckpointCallback(
                    {"val_logloss": "validation-mlogloss", "map_at_3": "validation-map_at_3"}
                )
            ]
        )
        map3_scores.append(bst.best_score)
        best_iteration = bst.best_iteration
        logloss_scores.append(evals_result['validation']['mlogloss'][best_iteration])

    tune.report({
        "map_at_3": np.mean(map3_scores),
        "val_logloss": np.mean(logloss_scores),
        "map_at_3_std": np.std(map3_scores)
    })


# --- Main Orchestration ---
def main():
    """Main function to run the full pipeline."""
    # --- 1. Load and Preprocess Data ---
    # NOTE: Update these paths to your actual data locations
    try:
        x_processed, y_encoded, target_encoder, preprocessor = load_and_preprocess_data(
            '../data/train.csv',
            '../data/Fertilizer_Prediction.csv'
        )
    except FileNotFoundError:
        logging.error("Data files not found. Please update the paths in main(). Exiting.")
        return

    # --- 2. Initialize Ray ---
    if not ray.is_initialized():
        ray.init(include_dashboard=False)

    # IMPROVEMENT: Put data into the object store once.
    x_ref = ray.put(x_processed)
    y_ref = ray.put(y_encoded)

    # --- 3. Define Hyperparameter Search Space ---
    search_space = {
        "booster": tune.choice(["gbtree", "dart"]),
        "rate_drop": tune.uniform(0.05, 0.3),
        "skip_drop": tune.uniform(0.05, 0.5),
        "learning_rate": tune.loguniform(0.05, 0.2),
        "max_depth": tune.choice([6, 8, 10, 12]),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "reg_lambda": tune.loguniform(0.1, 10.0),
        "reg_alpha": tune.loguniform(1e-3, 1.0),
        "gamma": tune.loguniform(0.05, 5.0),
        "min_child_weight": tune.choice([1, 3, 5, 7]),
        "colsample_bylevel": tune.uniform(0.6, 1.0),
        "colsample_bynode": tune.uniform(0.6, 1.0),
    }

    # --- 4. Configure and Run the Tuner ---
    scheduler = ASHAScheduler(
        metric="map_at_3",
        mode="max",
        # CRITICAL FIX: Use the constant defined outside, not a 'config' object.
        max_t=NUM_BOOST_ROUND,
        grace_period=EARLY_STOPPING_ROUNDS + 5,  # Ensure at least one early stop can happen
        reduction_factor=2
    )

    resources = {"cpu": 36, "gpu": 0}
    if torch.cuda.is_available():
        resources = {"cpu": 12, "gpu": 1}
    logging.info(f"Using Ray Tune resources: {resources}")

    tuner = Tuner(
        tune.with_resources(
            # Pass the object references to the trainable
            tune.with_parameters(train_xgboost, x_data=x_ref, y_data=y_ref),
            resources=resources
        ),
        param_space=search_space,
        tune_config=TuneConfig(
            scheduler=scheduler,
            num_samples=TUNE_NUM_SAMPLES,
            search_alg=OptunaSearch(metric="map_at_3", mode="max")
        ),
        run_config=RunConfig(name="xgboost_map3_tune_v2")
    )

    results = tuner.fit()

    # --- 5. Analyze Results and Retrain Best Model ---
    best_result = results.get_best_result(metric="map_at_3", mode="max")
    if not best_result:
        logging.error("Tuning finished, but no best result was found.")
        ray.shutdown()
        return

    logging.info("\n--- Best Trial Found ---")
    logging.info(f"  Config: {best_result.config}")
    logging.info(f"  MAP@3: {best_result.metrics.get('map_at_3', 'N/A'):.4f}")
    logging.info(f"  Log Loss: {best_result.metrics.get('val_logloss', 'N/A'):.4f}")

    # IMPROVEMENT: Retrain the model on the full dataset with the best hyperparameters
    logging.info("\nRetraining model on full dataset with best hyperparameters...")
    best_params = {
        'objective': 'multi:softprob',
        'eval_metric': ['mlogloss', 'map@3'],  # Can evaluate multiple metrics
        'num_class': len(y_encoded.unique()),
        'booster': best_result.config.get('booster'),
        'rate_drop': best_result.config.get('rate_drop'),
        'skip_drop': best_result.config.get('skip_drop'),
        'eta': best_result.config.get('learning_rate'),
        'max_depth': best_result.config.get('max_depth'),
        'subsample': best_result.config.get('subsample'),
        'colsample_bytree': best_result.config.get('colsample_bytree'),
        'lambda': best_result.config.get('reg_lambda'),
        'alpha': best_result.config.get('reg_alpha'),
        'gamma': best_result.config.get('gamma'),
        'min_child_weight': best_result.config.get('min_child_weight'),
        'colsample_bylevel': best_result.config.get('colsample_bylevel'),
        'colsample_bynode': best_result.config.get('colsample_bynode'),
        'seed': 42,
        'device': 'cuda',
        'tree_method': 'hist'
    }

    dtrain_full = xgb.DMatrix(x_processed, label=y_encoded, feature_names=x_processed.columns.tolist())

    # We need to determine the optimal number of rounds.
    # The best_result doesn't directly give us the average best_iteration.
    # A common practice is to train for a bit longer than the average stopping point
    # or to re-run CV just to find the optimal number of rounds.
    # For simplicity, we'll train for a fixed large number with early stopping on the full data.
    # This requires a validation set. We'll make a small one here.
    from sklearn.model_selection import train_test_split
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        x_processed, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    dtrain_final = xgb.DMatrix(X_train_final, label=y_train_final)
    dval_final = xgb.DMatrix(X_val_final, label=y_val_final)

    final_model = xgb.train(
        best_params,
        dtrain_final,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval_final, 'validation')],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100
    )

    # IMPROVEMENT: Save the final model and the preprocessor
    model_filename = "final_xgboost_model.json"
    preprocessor_filename = "preprocessor.joblib"
    encoder_filename = "target_encoder.joblib"

    final_model.save_model(model_filename)

    joblib.dump(preprocessor, preprocessor_filename)
    joblib.dump(target_encoder, encoder_filename)

    logging.info(f"Final model saved to {model_filename}")
    logging.info(f"Preprocessor saved to {preprocessor_filename}")
    logging.info(f"Target encoder saved to {encoder_filename}")

    ray.shutdown()


if __name__ == "__main__":
    main()
