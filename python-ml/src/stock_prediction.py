
# File: stock_prediction.py
# 
# Original Authors: Bao Vo and Cheong Koo
# 
#  Original Dates: 14/07/2021 (v1); 19/07/2021 (v2); 02/07/2024 (v3)
# 
# -----------------------------------------------------------------------
#  Code adapted from:
#   Title: Predicting Stock Prices with Python (NeuralNine)
# 
# 
#    YouTube: https://www.youtube.com/watch?v=PuZY9q-aKLw
#   Reference repo (tutorials by x4nth055 / The Python Code): see LICENSE in repo
# 
# 
# ------------------------------------------------------------------------
#  Modifications: 
#   Modified by: Suyogya “Rex” Raj Joshi
# 
#   Date: 28/08/2025
# 
#   Summary of changes: Switched to yfinance download, adjusted train/test dates,
# 
#   clarified scaling/reshaping comments, tweaked LSTM stack & dropout, added
# 
#   plotting, etc. (list your real changes briefly)
# 
# ------------------------------------------------------------------------------------------------
# License:
# 
#   Retain original license/notice from any borrowed sources. If you publish this
# 
#   project, include the upstream LICENSE files and attribution in your README.

# File: ensemble_prediction.py
# Task 6: Ensemble Modeling Approach
# Combining ARIMA/SARIMA with Deep Learning Models (LSTM/GRU/RNN)
#
# This extends stock_prediction.py (v0.5) with ensemble methods

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# ORIGINAL FUNCTIONS FROM stock_prediction.py (unchanged)
# ============================================================================

def load_and_preprocess_data(company, start_date, end_date, test_size=0.2):
    data = yf.download(company, start=start_date, end=end_date, progress=False)
    data = data.ffill().bfill().dropna()
    data["OHLC_Avg"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4.0

    values = data["OHLC_Avg"].values.astype(np.float32)

    split_point = int(len(values) * (1 - test_size))
    train_vals, test_vals = values[:split_point], values[split_point:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1))
    test_scaled = scaler.transform(test_vals.reshape(-1, 1))

    return (train_scaled, test_scaled, scaler, data)


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return X[..., None], y


def create_dl_model(model_type, sequence_length, n_features=1,
                    layers=[64, 32], dropout_rate=0.2, learning_rate=1e-3):
    RNNmap = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}
    if model_type not in RNNmap:
        raise ValueError(f"Choose from {list(RNNmap.keys())}")
    RNN = RNNmap[model_type]

    model = Sequential()
    if len(layers) == 1:
        model.add(RNN(layers[0], input_shape=(sequence_length, n_features)))
        model.add(Dropout(dropout_rate))
    else:
        model.add(RNN(layers[0], return_sequences=True, input_shape=(sequence_length, n_features)))
        model.add(Dropout(dropout_rate))
        for units in layers[1:-1]:
            model.add(RNN(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(RNN(layers[-1]))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                             scaler, epochs=30, batch_size=32, verbose=0):
    cut = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-cut], y_train[:-cut]
    X_val, y_val = X_train[-cut:], y_train[-cut:]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size,
                     shuffle=False, verbose=verbose, callbacks=callbacks)

    y_pred = model.predict(X_test, verbose=0)

    y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred).ravel()

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    return {
        "history": hist,
        "pred": y_pred_inv,
        "true": y_true_inv,
        "mae": mae,
        "rmse": rmse,
        "final_loss": hist.history["loss"][-1],
        "final_val_loss": hist.history["val_loss"][-1]
    }


# ============================================================================
# NEW TASK 6 FUNCTIONS: STATISTICAL MODELS
# ============================================================================

def train_arima_model(train_data, test_data, order=(5, 1, 0)):
   
    try:
        # Fit ARIMA on training data
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        
        # Forecast for test period
        forecast = fitted_model.forecast(steps=len(test_data))
        
        return forecast, fitted_model
    except Exception as e:
        print(f"ARIMA failed with order {order}: {e}")
        return None, None


def train_sarima_model(train_data, test_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
 
    try:
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        
        forecast = fitted_model.forecast(steps=len(test_data))
        
        return forecast, fitted_model
    except Exception as e:
        print(f"SARIMA failed: {e}")
        return None, None


def train_random_forest_model(X_train, y_train, X_test, n_estimators=100, max_depth=10):
  
    try:
        # Flatten sequences for Random Forest (it doesn't handle 3D input)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                     random_state=42, n_jobs=-1)
        model.fit(X_train_flat, y_train)
        
        predictions = model.predict(X_test_flat)
        
        return predictions, model
    except Exception as e:
        print(f"Random Forest failed: {e}")
        return None, None


# ============================================================================
# ENSEMBLE COMBINATION STRATEGIES
# ============================================================================

def simple_average_ensemble(predictions_dict):
  
    all_preds = np.array(list(predictions_dict.values()))
    return np.mean(all_preds, axis=0)


def weighted_average_ensemble(predictions_dict, weights):
  
    all_preds = np.array(list(predictions_dict.values()))
    weights = np.array(weights).reshape(-1, 1)
    return np.sum(all_preds * weights, axis=0)


def median_ensemble(predictions_dict):
   
    all_preds = np.array(list(predictions_dict.values()))
    return np.median(all_preds, axis=0)


# ============================================================================
# TASK 6 MAIN ENSEMBLE WORKFLOW
# ============================================================================

def run_ensemble_experiments(company="AAPL", start_date="2020-01-01", 
                            end_date="2023-08-01", seq_len=60):
   
    
    print("="*80)
    print("TASK 6: ENSEMBLE MODELING APPROACH")
    print("="*80)
    
    # Load data
    print(f"\n1. Loading data for {company}...")
    train_scaled, test_scaled, scaler, raw_data = load_and_preprocess_data(
        company, start_date, end_date
    )
    
    # Prepare sequences for DL models
    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len)
    
    # Get unscaled data for statistical models
    split_point = int(len(raw_data) * 0.8)
    train_original = raw_data["OHLC_Avg"].values[:split_point]
    test_original = raw_data["OHLC_Avg"].values[split_point:]
    
    # Storage for predictions
    all_predictions = {}
    all_metrics = []
    
    # ========================================================================
    # 2. Train Deep Learning Models
    # ========================================================================
    print("\n2. Training Deep Learning Models...")
    
    dl_configs = [
        {"name": "LSTM", "type": "LSTM", "layers": [64, 32]},
        {"name": "GRU", "type": "GRU", "layers": [64, 32]},
        {"name": "RNN", "type": "RNN", "layers": [64]},
    ]
    
    for config in dl_configs:
        print(f"   Training {config['name']}...")
        model = create_dl_model(config['type'], seq_len, layers=config['layers'])
        result = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, scaler, 
            epochs=30, batch_size=32, verbose=0
        )
        
        all_predictions[config['name']] = result['pred']
        all_metrics.append({
            "Model": config['name'],
            "MAE": result['mae'],
            "RMSE": result['rmse']
        })
        print(f"   {config['name']} - MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
    
    # ========================================================================
    # 3. Train Statistical Models (ARIMA/SARIMA)
    # ========================================================================
    print("\n3. Training Statistical Models...")
    
    # ARIMA
    print("   Training ARIMA...")
    arima_forecast, arima_model = train_arima_model(
        train_original, test_original, order=(5, 1, 0)
    )
    
    if arima_forecast is not None:
        # Align with DL predictions (skip first seq_len points)
        arima_aligned = arima_forecast[seq_len:]
        all_predictions['ARIMA'] = arima_aligned
        
        mae_arima = mean_absolute_error(result['true'], arima_aligned)
        rmse_arima = np.sqrt(mean_squared_error(result['true'], arima_aligned))
        all_metrics.append({"Model": "ARIMA", "MAE": mae_arima, "RMSE": rmse_arima})
        print(f"   ARIMA - MAE: {mae_arima:.4f}, RMSE: {rmse_arima:.4f}")
    
    # SARIMA
    print("   Training SARIMA...")
    sarima_forecast, sarima_model = train_sarima_model(
        train_original, test_original, 
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
    )
    
    if sarima_forecast is not None:
        sarima_aligned = sarima_forecast[seq_len:]
        all_predictions['SARIMA'] = sarima_aligned
        
        mae_sarima = mean_absolute_error(result['true'], sarima_aligned)
        rmse_sarima = np.sqrt(mean_squared_error(result['true'], sarima_aligned))
        all_metrics.append({"Model": "SARIMA", "MAE": mae_sarima, "RMSE": rmse_sarima})
        print(f"   SARIMA - MAE: {mae_sarima:.4f}, RMSE: {rmse_sarima:.4f}")
    
    # ========================================================================
    # 4. Train Random Forest (Optional Additional Model)
    # ========================================================================
    print("\n4. Training Random Forest...")
    rf_pred, rf_model = train_random_forest_model(
        X_train, y_train, X_test, n_estimators=100, max_depth=10
    )
    
    if rf_pred is not None:
        rf_pred_inv = scaler.inverse_transform(rf_pred.reshape(-1, 1)).ravel()
        all_predictions['RandomForest'] = rf_pred_inv
        
        mae_rf = mean_absolute_error(result['true'], rf_pred_inv)
        rmse_rf = np.sqrt(mean_squared_error(result['true'], rf_pred_inv))
        all_metrics.append({"Model": "RandomForest", "MAE": mae_rf, "RMSE": rmse_rf})
        print(f"   Random Forest - MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")
    
    # ========================================================================
    # 5. Create Ensemble Combinations
    # ========================================================================
    print("\n5. Creating Ensemble Combinations...")
    
    y_true = result['true']
    
    # Simple Average Ensemble
    ensemble_avg = simple_average_ensemble(all_predictions)
    mae_avg = mean_absolute_error(y_true, ensemble_avg)
    rmse_avg = np.sqrt(mean_squared_error(y_true, ensemble_avg))
    all_metrics.append({"Model": "Ensemble_Average", "MAE": mae_avg, "RMSE": rmse_avg})
    print(f"   Simple Average Ensemble - MAE: {mae_avg:.4f}, RMSE: {rmse_avg:.4f}")
    
    # Weighted Ensemble (inverse MAE as weights)
    metrics_df = pd.DataFrame(all_metrics[:-1])  # Exclude ensemble itself
    inverse_mae = 1.0 / metrics_df['MAE'].values
    weights = inverse_mae / inverse_mae.sum()
    
    ensemble_weighted = weighted_average_ensemble(all_predictions, weights)
    mae_weighted = mean_absolute_error(y_true, ensemble_weighted)
    rmse_weighted = np.sqrt(mean_squared_error(y_true, ensemble_weighted))
    all_metrics.append({"Model": "Ensemble_Weighted", "MAE": mae_weighted, "RMSE": rmse_weighted})
    print(f"   Weighted Ensemble - MAE: {mae_weighted:.4f}, RMSE: {rmse_weighted:.4f}")
    
    # Median Ensemble
    ensemble_median = median_ensemble(all_predictions)
    mae_median = mean_absolute_error(y_true, ensemble_median)
    rmse_median = np.sqrt(mean_squared_error(y_true, ensemble_median))
    all_metrics.append({"Model": "Ensemble_Median", "MAE": mae_median, "RMSE": rmse_median})
    print(f"   Median Ensemble - MAE: {mae_median:.4f}, RMSE: {rmse_median:.4f}")
    
    # ========================================================================
    # 6. Display Results
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(all_metrics).sort_values("MAE").reset_index(drop=True)
    print("\n" + results_df.to_string(index=False))
    
    # ========================================================================
    # 7. Visualization
    # ========================================================================
    print("\n6. Generating visualizations...")
    
    # Plot 1: All individual models + ensembles
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Individual DL models
    axes[0, 0].plot(y_true, label='Actual', linewidth=2, color='black')
    for name in ['LSTM', 'GRU', 'RNN']:
        if name in all_predictions:
            axes[0, 0].plot(all_predictions[name], label=name, alpha=0.7)
    axes[0, 0].set_title('Deep Learning Models')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Statistical models
    axes[0, 1].plot(y_true, label='Actual', linewidth=2, color='black')
    for name in ['ARIMA', 'SARIMA']:
        if name in all_predictions:
            axes[0, 1].plot(all_predictions[name], label=name, alpha=0.7)
    axes[0, 1].set_title('Statistical Models')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ensemble comparisons
    axes[1, 0].plot(y_true, label='Actual', linewidth=2, color='black')
    axes[1, 0].plot(ensemble_avg, label='Average Ensemble', alpha=0.7)
    axes[1, 0].plot(ensemble_weighted, label='Weighted Ensemble', alpha=0.7)
    axes[1, 0].plot(ensemble_median, label='Median Ensemble', alpha=0.7)
    axes[1, 0].set_title('Ensemble Methods Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE comparison bar chart
    axes[1, 1].barh(results_df['Model'], results_df['MAE'])
    axes[1, 1].set_xlabel('Mean Absolute Error (MAE)')
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('artifacts/ensemble_results.png', dpi=300, bbox_inches='tight')
    print("   Saved: artifacts/ensemble_results.png")
    plt.show()
    
    # ========================================================================
    # 8. Save Results
    # ========================================================================
    os.makedirs("artifacts", exist_ok=True)
    
    results_df.to_csv("artifacts/ensemble_metrics.csv", index=False)
    print("   Saved: artifacts/ensemble_metrics.csv")
    
    # Save predictions
    pred_df = pd.DataFrame({"Actual": y_true})
    for name, pred in all_predictions.items():
        pred_df[name] = pred
    pred_df['Ensemble_Average'] = ensemble_avg
    pred_df['Ensemble_Weighted'] = ensemble_weighted
    pred_df['Ensemble_Median'] = ensemble_median
    pred_df.to_csv("artifacts/ensemble_predictions.csv", index=False)
    print("   Saved: artifacts/ensemble_predictions.csv")
    
    return results_df, all_predictions, ensemble_weighted


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results_df, predictions, best_ensemble = run_ensemble_experiments(
        company="AAPL",
        start_date="2020-01-01",
        end_date="2023-08-01",
        seq_len=60
    )
    
    print("\n" + "="*80)
    print("Task 6 Complete! Check 'artifacts/' folder for saved results.")
    print("="*80)