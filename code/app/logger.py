import csv
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "usage_log.csv")

def log_interaction(ticker, recommendation, explanation, latency_ms):
    """
    Append a log entry to the CSV file for each analysis run.
    """
    fields = [datetime.now().isoformat(), ticker, recommendation, explanation, latency_ms]

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticker", "recommendation", "explanation", "latency_ms"])
        writer.writerow(fields)

if __name__ == "__main__":
    # Simple test
    log_interaction("MSFT", "BUY", "Test explanation", 123)
