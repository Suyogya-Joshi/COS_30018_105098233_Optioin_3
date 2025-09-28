
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

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# load stock data and preprocess
def load_and_preprocess_data(company, start_date, end_date, test_size=0.2):
    data = yf.download(company, start=start_date, end=end_date, progress=False)
    data = data.ffill().bfill().dropna()
    data["OHLC_Avg"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4.0

    values = data["OHLC_Avg"].values.astype(np.float32)

    # split train and test
    split_point = int(len(values) * (1 - test_size))
    train_vals, test_vals = values[:split_point], values[split_point:]

    # scale based on train only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1))
    test_scaled = scaler.transform(test_vals.reshape(-1, 1))

    return (train_scaled, test_scaled, scaler, data)


# create input sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return X[..., None], y


# split into train and test
def split_data(X, y, test_size=0.2):
    split_point = int(len(X) * (1 - test_size))
    return X[:split_point], X[split_point:], y[:split_point], y[split_point:]


# build the model
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


# train the model and test it
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                             scaler, epochs=30, batch_size=32, verbose=0):
    # take last 10% of train as validation
    cut = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-cut], y_train[:-cut]
    X_val, y_val = X_train[-cut:], y_train[-cut:]

    # use callbacks to stop early and reduce LR
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size,
                     shuffle=False, verbose=verbose, callbacks=callbacks)

    # predict
    y_pred = model.predict(X_test, verbose=0)

    # bring back to original scale
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


# run a few experiments (Task 4)
def run_experiments(X_train, y_train, X_test, y_test, sequence_length, scaler):
    configs = [
        {"model_type": "LSTM", "layers": [64], "epochs": 30, "batch_size": 32},
        {"model_type": "GRU",  "layers": [64], "epochs": 30, "batch_size": 32},
        {"model_type": "RNN",  "layers": [64], "epochs": 30, "batch_size": 32},
        {"model_type": "LSTM", "layers": [64, 32], "epochs": 40, "batch_size": 32},
    ]
    results = []
    for cfg in configs:
        try:
            model = create_dl_model(cfg["model_type"], sequence_length, layers=cfg["layers"])
            res = train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                                           scaler, epochs=cfg["epochs"], batch_size=cfg["batch_size"])
            results.append({
                "Model_Type": cfg["model_type"],
                "Layers": str(cfg["layers"]),
                "Epochs": cfg["epochs"],
                "Batch_Size": cfg["batch_size"],
                "MAE": res["mae"],
                "RMSE": res["rmse"],
                "Final_Loss": res["final_loss"],
                "Final_Val_Loss": res["final_val_loss"]
            })
        except Exception as e:
            print("Failed:", e)
    return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)


# show results
def plot_results_comparison(results_df):
    print("\n=== Task 4 Results ===")
    print(results_df.round(4).to_string(index=False))


# plot predictions
def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()


# main function (Task 4 + Task 5)
def main():
    COMPANY = "AAPL"
    START, END = "2020-01-01", "2023-08-01"
    SEQ_LEN = 60

    # load data (only once)
    train_scaled, test_scaled, scaler, _ = load_and_preprocess_data(COMPANY, START, END)

    # make sequences
    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
    X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

    # run experiments
    results_df = run_experiments(X_train, y_train, X_test, y_test, SEQ_LEN, scaler)
    plot_results_comparison(results_df)

    # pick best
    best = results_df.iloc[0]
    model = create_dl_model(best["Model_Type"], SEQ_LEN, layers=eval(best["Layers"]))
    res = train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                                   scaler, epochs=int(best["Epochs"]), batch_size=int(best["Batch_Size"]), verbose=1)

    print(f"\nBest model ({best['Model_Type']} {best['Layers']}) "
          f"> MAE: {res['mae']:.4f}, RMSE: {res['rmse']:.4f}")

    # plot predictions for best model
    plot_predictions(res["true"], res["pred"], f"Best {best['Model_Type']} {best['Layers']}")

    # save model and predictions
    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/best_model.keras")
    pd.DataFrame({"actual": res["true"], "predicted": res["pred"]}).to_csv("artifacts/predictions.csv", index=False)

    return results_df, model, scaler


if __name__ == "__main__":
    results, best_model, scaler = main()
