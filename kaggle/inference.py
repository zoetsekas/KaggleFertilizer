import os

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from kaggle import add_features, FertilizerClassifier
import torch.nn.functional as F

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

if __name__ == '__main__':
    inference_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    inference_df = add_features(inference_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(feature_range=(-1, 1)), NUMERICAL_COLS),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
        ],
        remainder='drop'  # Drop any columns not explicitly transformed (like 'id' if it wasn't dropped already)
    )

    X_inference = preprocessor.transform(inference_df)
    inference_tensor = torch.tensor(X_inference.toarray() if hasattr(X_inference, 'toarray') else X_inference,
                                    dtype=torch.float32).to(device)

    best_result = None
    if best_result.path:
        print(f"Best trial checkpoint path: {best_result.path}")
        # You would load the model from this checkpoint for inference
        loaded_model = FertilizerClassifier(num_classes=7, activation_fn_name='elu', dropout_rate=0.1820, input_size=35,
                                            l1_units=256, l2_units=64)
        loaded_model.load_state_dict(torch.load(os.path.join(best_result.path, "model.pt")))

        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(inference_tensor)
            probabilities = F.softmax(outputs, dim=1)

        top_k = 3
        predictions_list = []  # To store the formatted predictions for each sample

        print(f"\n--- Top {top_k} Fertilizer Predictions (for display and CSV) ---")
        for i in range(inference_df.shape[0]):
            row_probabilities = probabilities[i]
            top_prob, top_indices = torch.topk(row_probabilities, top_k)

            # Convert indices to fertilizer names
            predicted_fertilizers = [label_encoder.inverse_transform([idx.item()])[0] for idx in top_indices]

            # Join the predicted fertilizer names into a single string
            joined_predictions = " ".join(predicted_fertilizers)

            predictions_list.append({"Fertilizer Name": joined_predictions})

            # Optional: Print for immediate feedback
            print(f"\nSample {i + 1} (Input: {inference_df.iloc[i].to_dict()}):")
            print(f"  Predicted Fertilizers (Top {top_k}): {joined_predictions}")
            for j in range(top_k):
                print(f"    - {predicted_fertilizers[j]}: {top_prob[j].item():.4f}")