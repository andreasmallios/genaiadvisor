import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import joblib
from pathlib import Path

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Directory setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def fetch_ticker_data(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{ticker}_10y.csv"
    if csv_path.exists():
        print(f"[INFO] Loading cached data for {ticker} from {csv_path}")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    else:
        print(f"[INFO] Fetching {period} data for {ticker} from Yahoo Finance")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker}")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.name = "Date"
        df.to_csv(csv_path)
        print(f"[INFO] Saved data for {ticker} to {csv_path}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['SMA_50'] = df_feat['Close'].rolling(window=50).mean()
    df_feat['SMA_diff'] = df_feat['SMA_10'] - df_feat['SMA_50']

    delta = df_feat['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    df_feat['EMA_12'] = df_feat['Close'].ewm(span=12, adjust=False).mean()
    df_feat['EMA_26'] = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = df_feat['EMA_12'] - df_feat['EMA_26']

    df_feat['Return_1d'] = df_feat['Close'].pct_change(1)
    df_feat['Return_5d'] = df_feat['Close'].pct_change(5)
    df_feat['Return_10d'] = df_feat['Close'].pct_change(10)

    df_feat.dropna(inplace=True)
    return df_feat


def create_labels(df: pd.DataFrame, threshold=0.02) -> pd.Series:
    future_returns = df['Close'].shift(-10) / df['Close'] - 1
    labels = np.where(future_returns > threshold, 2, np.where(future_returns < -threshold, 0, 1))
    return pd.Series(labels, index=df.index)


def build_tf_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_tf_model(ticker_list, threshold=0.02):
    all_features = []
    all_labels = []

    for ticker in ticker_list:
        try:
            print(f"[INFO] Processing {ticker}")
            df = fetch_ticker_data(ticker)
            df_feat = compute_features(df)
            labels = create_labels(df_feat, threshold=threshold)

            features = df_feat[['SMA_10', 'SMA_50', 'SMA_diff', 'RSI', 'MACD',
                                'Return_1d', 'Return_5d', 'Return_10d', 'Volume']]

            combined = pd.concat([features, labels.rename("Label")], axis=1).dropna()
            all_features.append(combined.drop("Label", axis=1))
            all_labels.append(combined["Label"])
        except Exception as e:
            print(f"[WARN] Skipping {ticker} due to: {e}")

    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    # ✅ Log label distribution
    print("[INFO] Label distribution:\n", y.value_counts())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # ✅ Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("[INFO] Computed class weights:", class_weights_dict)

    model = build_tf_model(input_dim=X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # ✅ Use class weights in training
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[es],
        class_weight=class_weights_dict
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"[INFO] Test Accuracy: {test_acc:.2%}")

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model.save(MODEL_DIR / "tf_model.h5")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"[INFO] TensorFlow model and scaler saved to {MODEL_DIR}")


def predict_signal(df: pd.DataFrame) -> dict:
    from datetime import datetime
    features = compute_features(df).iloc[[-1]][['SMA_10', 'SMA_50', 'SMA_diff', 'RSI', 'MACD',
                                                'Return_1d', 'Return_5d', 'Return_10d', 'Volume']]
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    features_scaled = scaler.transform(features)

    model = load_model(MODEL_DIR / "tf_model.h5")
    pred_probs = model.predict(features_scaled)[0]
    pred_class = np.argmax(pred_probs)

    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
    recommendation = mapping[pred_class]
    reason = f"TensorFlow model prediction with confidence: {pred_probs[pred_class]:.2%}"

    return {
        "signal": "ML Classifier (TF)",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(datetime.now().date())
    }


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[2] / "data" / "tickers.csv"
    tickers_df = pd.read_csv(csv_path)
    tickers = tickers_df["Symbol"].dropna().unique().tolist()

    train_and_save_tf_model(tickers)