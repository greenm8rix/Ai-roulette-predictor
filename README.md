# Roulette Predictor & Automaton

## 1. Description

This project is a Python-based application designed to predict roulette wheel outcomes using a TensorFlow/Keras LSTM model with an attention mechanism. It also includes a Selenium-based web scraper to interact with an online roulette game, extract live numbers, feed them to the predictor, and (conceptually) automate betting based on a Fibonacci strategy.

The application is structured into several modules:
- `main.py`: Main entry point for the application.
- `config.py`: Handles all configurations (model parameters, file paths, web scraping URLs, credentials, logging).
- `model_utils.py`: Contains utility components for the TensorFlow model, like the custom `Attention` layer.
- `wheel_predictor.py`: Implements the `WheelPredictor` class, managing the ML model's logic, training, state, and predictions.
- `web_scraper.py`: Implements the `RouletteScraper` class, managing Selenium browser interactions, data extraction, and integration with the `WheelPredictor`.

## 2. Prerequisites

- Python 3.8+
- pip (Python package installer)
- Google Chrome or Brave Browser installed.
- ChromeDriver (or Brave-specific WebDriver) compatible with your browser version.

## 3. Setup Instructions

### a. Clone the Repository (if applicable)
```bash
git clone <repository-url>
cd <repository-directory>
```

### b. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```
Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### c. Install Dependencies
First, ensure you have the necessary packages. Create a `requirements.txt` file with the following content:

```txt
tensorflow
numpy
scikit-learn
selenium
keyboard
# Add any other specific versions if needed, e.g., tensorflow==2.10.0
```

Then, install the dependencies:
```bash
pip install -r requirements.txt
```

### d. WebDriver Setup
1.  Download the ChromeDriver (or Brave WebDriver) that matches your installed browser version.
    - ChromeDriver: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
2.  Place the executable (e.g., `chromedriver.exe`) in a known location on your system.

## 4. Configuration

Open the `config.py` file and update the following sections as needed:

### a. `WEB_CONFIG`:
   - **`paths`**:
     - `brave_driver`: Set the absolute path to your `chromedriver.exe` (or equivalent for Brave).
       Example: `r"C:\path\to\your\chromedriver.exe"`
     - `brave_browser`: Set the absolute path to your Chrome/Brave browser executable.
       Example: `r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"`
   - **`credentials`**:
     - `username`: Your login username for the target website.
     - `password`: Your login password.
       **Security Note**: For better security, consider using environment variables or a dedicated secrets management tool instead of hardcoding credentials.
   - **`urls`**:
     - `base`: The base URL of the lottery/casino website.
     - `iframe_pattern`: A string pattern unique to the URL of the iframe containing the roulette game. This is used to identify and switch to the correct iframe.

### b. `CONFIG` (Model Configuration):
   - Review parameters like `window_size`, `prediction_size`, `model_file`, `state_file`, LSTM units, dropout rates, etc. Default values are provided.
   - `model_file`: Path where the trained Keras model will be saved/loaded.
   - `state_file`: Path where the predictor's state (training data, balance) will be saved/loaded.

### c. `LOGGING_CONFIG`:
   - Adjust logging level and format if desired.

## 5. Running the Application
Once setup and configuration are complete, run the application from the project's root directory:
```bash
python main.py
```
The application will:
1. Initialize the `WheelPredictor` (loading or creating a model and state).
2. Initialize the `RouletteScraper`.
3. Launch the browser, navigate to the specified URL, and attempt to log in.
4. Try to find and switch to the game iframe.
5. Begin collecting initial roulette numbers.
6. Enter the main loop:
   - Extract new numbers.
   - Use the `WheelPredictor` to get predictions.
   - Simulate betting and update balance (as per `WheelPredictor` logic).
   - Add new data to the `WheelPredictor` for training.
   - Periodically retrain the model.

To stop the bot, you can usually press `Ctrl+C` in the terminal where it's running, or `ESC` if the `keyboard` library's hook is active during certain phases (like initial data collection).

## 6. Notes

- **Betting Strategy**: The current implementation uses a Fibonacci sequence for bet sizing based on losing streaks. The actual betting on the website (placing chips) is not fully implemented in the provided `web_scraper.py`'s JavaScript execution part; it focuses on number extraction and prediction. The `WheelPredictor` simulates balance updates based on whether the outcome falls within its predicted section.
- **Model Training**: The model trains periodically based on the accumulated data. The effectiveness of the predictions will depend heavily on the quality and quantity of data, as well as the model architecture and hyperparameters.
- **Ethical Considerations**: This tool interacts with online gambling sites. Be aware of the terms of service of any website you use this on and the legal regulations regarding automated play in your jurisdiction. Gambling can be addictive; please play responsibly.
- **Error Handling**: The application includes logging for errors. Check the console output and log files (if configured) for troubleshooting.

## 7. `.gitignore`
It's recommended to use a `.gitignore` file to exclude unnecessary files from version control. Create a file named `.gitignore` in the root of your project with content like:
```
# Python
__pycache__/
*.py[cod]
*$py.class

# Keras/TensorFlow model and state files
# (Uncomment if you don't want to commit them, especially if large)
# wheel_model.keras
# predictor_state.pkl

# Virtual environment
venv/
env/
.venv/
.env/
pip-selfcheck.json

# IDE specific
.vscode/
.idea/

# OS specific
.DS_Store
Thumbs.db
