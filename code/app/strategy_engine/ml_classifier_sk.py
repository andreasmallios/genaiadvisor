import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

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
    labels = np.where(future_returns > threshold, 1, np.where(future_returns < -threshold, -1, 0))
    return pd.Series(labels, index=df.index)

def train_and_save_models(ticker="MSFT"):
    print(f"[INFO] Fetching and preparing data for {ticker}")
    df = fetch_ticker_data(ticker)
    df_feat = compute_features(df)
    labels = create_labels(df_feat)

    features = df_feat[['SMA_10', 'SMA_50', 'SMA_diff', 'RSI', 'MACD',
                        'Return_1d', 'Return_5d', 'Return_10d', 'Volume']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels,
                                                        test_size=0.2, random_state=42, shuffle=False)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("[INFO] Random Forest Classifier:")
    print(classification_report(y_test, y_pred_rf))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("[INFO] Logistic Regression:")
    print(classification_report(y_test, y_pred_lr))

    joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
    joblib.dump(lr, MODEL_DIR / "lr_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"[INFO] Models and scaler saved to {MODEL_DIR}")

def train_and_save_models_multi(ticker_list, threshold=0.02):
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

            # Drop any rows where labels are missing
            combined = pd.concat([features, labels.rename("Label")], axis=1).dropna()
            all_features.append(combined.drop("Label", axis=1))
            all_labels.append(combined["Label"])
        except Exception as e:
            print(f"[WARN] Skipping {ticker} due to: {e}")

    # Combine and scale features
    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # -----------------------------
    # Random Forest - Grid Search
    # -----------------------------
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf_base = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_

    print("[INFO] Random Forest Classifier:")
    print("Best Parameters:", rf_grid.best_params_)
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

    # -----------------------------
    # Logistic Regression - Grid Search
    # -----------------------------
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }

    lr_base = LogisticRegression(max_iter=1000)
    lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    lr = lr_grid.best_estimator_

    print("[INFO] Logistic Regression:")
    print("Best Parameters:", lr_grid.best_params_)
    y_pred_lr = lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))

    # -----------------------------
    # Save models
    # -----------------------------
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
    joblib.dump(lr, MODEL_DIR / "lr_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"[INFO] Models and scaler saved to {MODEL_DIR}")


def predict_signal(df: pd.DataFrame) -> dict:
    from datetime import datetime
    features = compute_features(df).iloc[[-1]][['SMA_10', 'SMA_50', 'SMA_diff', 'RSI', 'MACD',
                                                'Return_1d', 'Return_5d', 'Return_10d', 'Volume']]
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    features_scaled = scaler.transform(features)

    rf = joblib.load(MODEL_DIR / "rf_model.pkl")
    pred_rf = rf.predict(features_scaled)[0]

    lr = joblib.load(MODEL_DIR / "lr_model.pkl")
    pred_lr = lr.predict(features_scaled)[0]

    pred_sum = pred_rf + pred_lr
    if pred_sum >= 1:
        recommendation = "BUY"
        reason = "ML classifiers indicate a BUY condition based on historical patterns."
    elif pred_sum <= -1:
        recommendation = "SELL"
        reason = "ML classifiers indicate a SELL condition based on historical patterns."
    else:
        recommendation = "HOLD"
        reason = "ML classifiers do not indicate a strong BUY or SELL condition."

    return {
        "signal": "ML Classifier",
        "recommendation": recommendation,
        "reason": reason,
        "date": str(datetime.now().date())
    }

if __name__ == "__main__":
    import pandas as pd

    # Path relative to the project root
    csv_path = Path(__file__).resolve().parents[2] / "data" / "tickers.csv"
    tickers_df = pd.read_csv(csv_path)

    # Use the 'Symbol' column
    tickers = tickers_df["Symbol"].dropna().unique().tolist()

    train_and_save_models_multi(tickers)
