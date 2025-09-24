from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os


# Baseline Linear Regression

def train_baseline_model(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Save predictions
    predictions = pd.DataFrame(
        {
            'datetime': df['datetime'].iloc[-len(y_test):],
            'actual': y_test,
            'predicted': y_pred,
        }
    )

    # Resolve save path to project root outputs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(project_root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    predictions_path = os.path.join(outputs_dir, 'predictions.csv')

    predictions.to_csv(predictions_path, index=False)

    return model
