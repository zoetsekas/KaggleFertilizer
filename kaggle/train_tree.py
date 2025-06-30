import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from ray import tune, ray
from ray.tune import RunConfig, TuneConfig, Tuner
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from kaggle import add_features

from kaggle.data_processing import process_data

# 1. (New) Create a helper function with the core metric logic.
# This function works with standard NumPy arrays and is easy to reuse.
def _calculate_map_at_3(y_true_labels: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Core logic for calculating Mean Average Precision at 3 (MAP@3)."""
    # Ensure y_true is integer type for comparison
    y_true_labels = y_true_labels.astype(int)

    # Reshape prediction array if it's flattened (as it is during xgb.train)
    if y_pred_probs.ndim == 1:
        num_samples = len(y_true_labels)
        if num_samples == 0:
            return 0.0
        num_classes = len(y_pred_probs) // num_samples
        y_pred_probs = y_pred_probs.reshape(num_samples, num_classes)

    # Get the indices of the top 3 predictions by sorting probabilities
    top3_preds_indices = np.argsort(y_pred_probs, axis=1)[:, -3:]

    total_ap_at_3 = 0.0
    for i in range(len(y_true_labels)):
        true_label = y_true_labels[i]
        # Predictions are sorted ascending, so reverse to get descending order
        predicted_top3_for_sample = top3_preds_indices[i][::-1]

        ap_at_3_for_sample = 0.0
        if true_label in predicted_top3_for_sample:
            # Find the 1-based rank of the true label in the top 3 predictions
            rank = np.where(predicted_top3_for_sample == true_label)[0][0] + 1
            ap_at_3_for_sample = 1.0 / rank
        total_ap_at_3 += ap_at_3_for_sample

    return total_ap_at_3 / len(y_true_labels) if len(y_true_labels) > 0 else 0.0




# 2. Update map_at_3 to be a wrapper that conforms to XGBoost's `feval` signature.
def map_at_3(y_pred: np.ndarray, y_true: xgb.DMatrix) -> list[tuple[str, float]]:
    """
    XGBoost `feval` compatible wrapper for MAP@3 calculation.
    Signature: (predictions, DMatrix_with_labels)
    """
    true_labels = y_true.get_label()
    map_score = _calculate_map_at_3(true_labels, y_pred)
    # FIX: Use a compliant metric name
    return [('map_at_3', map_score)]


# 3. Define the training function for XGBoost
# 3. Define the training function for XGBoost with Stratified K-Fold CV
# In train_tree.py

def train_xgboost(config, x_data, y_data):
    """
    Trains a multi-class XGBoost classifier using Stratified K-Fold cross-validation
    and reports the average metrics back to Ray Tune.
    """
    # Ensure y_data is a 1D numpy array for consistency
    if isinstance(y_data, (pd.DataFrame, pd.Series)):
        y_data = y_data.values.ravel()

    # --- StratifiedKFold Implementation ---
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    map3_scores = []
    logloss_scores = []

    num_class = len(np.unique(y_data))

    # In your train_xgboost function

    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': num_class,
        # --- NEW: Add the booster and its parameters ---
        'booster': config.get('booster'),
        'rate_drop': config.get('rate_drop'),
        'skip_drop': config.get('skip_drop'),
        # --- End of new parameters ---
        'eta': config.get('learning_rate'),
        'max_depth': config.get('max_depth'),
        'subsample': config.get('subsample'),
        'colsample_bytree': config.get('colsample_bytree'),
        'lambda': config.get('reg_lambda'),
        'alpha': config.get('reg_alpha'),
        'seed': 42,
        'gamma': config.get('gamma'),
        'min_child_weight': config.get('min_child_weight'),
        'colsample_bylevel': config.get('colsample_bylevel'),
        'colsample_bynode': config.get('colsample_bynode'),
        'device': 'cuda',
        'tree_method': 'hist'
    }

    # Loop over each fold
    for fold, (train_index, val_index) in enumerate(skf.split(x_data, y_data)):
        X_train, X_val = x_data.iloc[train_index], x_data.iloc[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # --- FIX: Correctly capture evaluation history ---
        # 1. Initialize an empty dictionary to hold the evaluation results.
        evals_result = {}

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=config.get('num_boost_round', 100_000),
            evals=[(dval, 'validation')],
            # 2. Pass the dictionary here. XGBoost will populate it during training.
            evals_result=evals_result,
            custom_metric=map_at_3,
            verbose_eval=False,
            early_stopping_rounds=20,
            callbacks=[
                TuneReportCheckpointCallback(
                    {
                        "val_logloss": "validation-mlogloss",
                        # FIX: Use a compliant metric name
                        "map_at_3": "validation-map_at_3"
                    }
                )
            ]
        )

        # After early stopping, bst.best_score gives the best validation metric
        map3_scores.append(bst.best_score)

        # 3. Access the populated dictionary to get the logloss at the best iteration.
        best_iteration = bst.best_iteration
        logloss_scores.append(evals_result['validation']['mlogloss'][best_iteration])

    # --- Report Average Metrics to Ray Tune ---
    avg_map3 = np.mean(map3_scores)
    avg_logloss = np.mean(logloss_scores)
    std_map3 = np.std(map3_scores)

    # FIX: Use a compliant metric name
    tune.report({
        "map_at_3": avg_map3,
        "val_logloss": avg_logloss,
        "map_at_3_std": std_map3
    })



# ... (The rest of your tune_xgboost and __main__ block remains the same) ...
def tune_xgboost(x_final_for_tune: pd.DataFrame, y_encoded: pd.DataFrame):
    if not ray.is_initialized():
        ray.init(include_dashboard=False)

    # 3. Define the hyperparameter search space
        # In your tune_xgboost function

    search_space = {
        # --- NEW: Let Tune choose the booster ---
        "booster": tune.choice(["gbtree", "dart"]),

        # --- DART-specific parameters (ignored if booster is 'gbtree') ---
        # Fraction of trees to drop
        "rate_drop": tune.uniform(0.05, 0.3),
        # Probability of skipping the dropout procedure for a round
        "skip_drop": tune.uniform(0.05, 0.5),

        # --- Existing Parameters (can be slightly widened for the new booster) ---
        "learning_rate": tune.loguniform(0.05, 0.2),
        "max_depth": tune.choice([6, 8, 10, 12]),  # DART often works well with slightly shallower trees
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "reg_lambda": tune.loguniform(0.1, 10.0),
        "reg_alpha": tune.loguniform(1e-3, 1.0),
        "gamma": tune.loguniform(0.05, 5.0),
        "min_child_weight": tune.choice([1, 3, 5, 7]),
        "num_boost_round": 100_000,
        "colsample_bylevel": tune.uniform(0.6, 1.0),
        "colsample_bynode": tune.uniform(0.6, 1.0),
    }

    # 4. Configure the tuning experiment
    # Update the scheduler to monitor and maximize our new MAP@3 metric
    # In your tune_xgboost function

    scheduler = ASHAScheduler(
        # FIX: Use a compliant metric name
        metric="map_at_3",
        mode="max",
        # BUGFIX: `config` is not defined here. Use the static value from search_space.
        max_t=search_space["num_boost_round"],
        grace_period=50,
        reduction_factor=2
    )

    resources = {"cpu": 36, "gpu": 0}
    if torch.cuda.is_available():
        resources = {"cpu": 12, "gpu": 1}

    print(f"Using resources: {resources}")

    # Define the Tuner
    tuner = Tuner(
        tune.with_resources(
            tune.with_parameters(trainable=train_xgboost, x_data=x_final_for_tune, y_data=y_encoded),
            resources=resources),
        param_space=search_space,
        tune_config=TuneConfig(
            scheduler=scheduler,
            num_samples=20,  # Increased samples for a more thorough search
            # --- Add the search algorithm here ---
            search_alg=OptunaSearch(
                # FIX: Use a compliant metric name
                metric="map_at_3",
                mode="max"
            )
        ),
        run_config=RunConfig(
            name="xgboost_map3_tune",
            callbacks=[TBXLoggerCallback()],
        )
    )

    # Run the tuning experiment
    results = tuner.fit()

    # Analyze and print the best results
    # FIX: Use a compliant metric name
    best_result = results.get_best_result(metric="map_at_3", mode="max")

    print("\nBest trial found:")
    print(f"  Config: {best_result.config}")

    metrics = best_result.metrics
    # FIX: Use a compliant metric name
    map_score = metrics.get("map_at_3")
    logloss = metrics.get("val_logloss")
    accuracy = metrics.get("val_accuracy")

    print(f"  MAP@3: {map_score:.4f}" if map_score is not None else "  MAP@3: N/A")
    print(f"  Validation Log Loss: {logloss:.4f}" if logloss is not None else "  Validation Log Loss: N/A")
    print(f"  Validation Accuracy: {accuracy:.4f}" if accuracy is not None else "  Validation Accuracy: N/A")

    ray.shutdown()


# Example of how you might call this function
if __name__ == "__main__":
    # You would load your preprocessed X_final_for_tune (DataFrame) and y_encoded (NumPy array) here
    # from your main data preparation script.
    # x, y = process_data()
    df1 = pd.read_csv('../data/train.csv')
    df2 = pd.read_csv('../data/Fertilizer_Prediction.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop(columns=['id'])

    TARGET_COLUMN = 'Fertilizer Name'
    # Preprocessing
    df = add_features(df)
    x = df.drop(columns=[TARGET_COLUMN])
    CATEGORICAL_COLS = list(df.select_dtypes(exclude=np.number).columns)
    if TARGET_COLUMN in CATEGORICAL_COLS: CATEGORICAL_COLS.remove(TARGET_COLUMN)
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(handle_unknown='error'), CATEGORICAL_COLS)
        ], remainder='passthrough')
    x_processed = preprocessor.fit_transform(x)
    feature_names = list(preprocessor.get_feature_names_out())
    x = pd.DataFrame(x_processed, columns=feature_names)

    y_labels = df[TARGET_COLUMN]

    # --- Target Encoding ---
    # Encode the target variable 'Fertilizer Name' into numerical labels
    target_label_encoder = LabelEncoder()
    y = pd.Series(target_label_encoder.fit_transform(y_labels), name=TARGET_COLUMN)

    tune_xgboost(x, y)