# -----------------------------
# Main Configuration
# -----------------------------
CONFIG = {
    "wheel_order": [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5,
        24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ],
    "window_size": 13,
    "prediction_size": 10,
    "balance": 1000,
    "training_deque_maxlen": 2000,
    "model_file": "wheel_model.keras",
    "state_file": "predictor_state.pkl",
    "data_augmentation": True,  # Enable rotational augmentation
    "lstm_units1": 128,
    "lstm_units2": 64,
    "bidirectional": True,
    "dropout_rate1": 0.3,
    "dropout_rate2": 0.2,
    "dense_units": 64,
    "l2_reg": 1e-4,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "min_training_samples": 50,
    "augmentation_rotations": 3  # Number of rotational augmentations per sample
}

WHEEL_ORDER = CONFIG["wheel_order"]
NUM_TO_POS = {num: idx for idx, num in enumerate(WHEEL_ORDER)}

# -----------------------------
# Web Scraper Configuration
# -----------------------------
WEB_CONFIG = {
    "paths": {
        "brave_driver": r"C:\Users\nawaf\Videos\chromedriver-win64\chromedriver.exe",
        "brave_browser": r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
    },
    "credentials": {
        "username": "nawafsheikh10@gmail.com",
        "password": "Sheikh123!"  # Consider using environment variables for sensitive data
    },
    "urls": {
        "base": "https://lottery.mt/",
        "iframe_pattern": "dragonara.evo-games.com/frontend/evo/r2"
    },
    "fibonacci_max_streak": 10, # Max losing streak for Fibonacci betting
    "data_collection_limit": 507 # Limit for historical data collection loop
}

# -----------------------------
# Logging Configuration
# -----------------------------
LOGGING_CONFIG = {
    "level": "INFO", # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}
