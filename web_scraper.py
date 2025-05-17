import time
import logging
import keyboard # Keep for potential manual interrupt, though not ideal for unattended scripts

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

from config import WEB_CONFIG, LOGGING_CONFIG, CONFIG as PRED_CONFIG # PRED_CONFIG for predictor related values
from wheel_predictor import WheelPredictor # Import the refactored predictor

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"]
)
logger = logging.getLogger(__name__)

class RouletteScraper:
    """
    Manages Selenium browser interactions for logging into a website,
    navigating to a game iframe, extracting roulette numbers, and
    integrating with WheelPredictor for predictions and betting.
    """
    def __init__(self, predictor_instance):
        self.predictor = predictor_instance
        self.driver = None
        self.wait = None
        self._setup_driver()
        self.losing_streak = 0
        self.current_bet_amount = 1 # Initial bet amount

    def _setup_driver(self):
        """Initializes the Selenium WebDriver."""
        try:
            options = webdriver.ChromeOptions()
            options.binary_location = WEB_CONFIG["paths"]["brave_browser"]
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--start-maximized") # Start maximized for better element visibility
            # options.add_argument("--headless") # Consider for running without UI
            # options.add_argument("--log-level=3") # Suppress console noise from Chrome/Driver

            service = Service(WEB_CONFIG["paths"]["brave_driver"])
            self.driver = webdriver.Chrome(service=service, options=options)
            self.wait = WebDriverWait(self.driver, 30) # Default wait time of 30 seconds
            logger.info("WebDriver initialized successfully.")
        except WebDriverException as e:
            logger.error(f"WebDriver initialization failed: {e}")
            raise  # Re-raise to stop execution if driver fails

    def _get_fibonacci_bet(self, streak):
        """Calculates Fibonacci bet amount based on losing streak."""
        if streak <= 0: return 1
        if streak == 1: return 1
        a, b = 1, 1
        for _ in range(2, streak + 1):
            a, b = b, a + b
        return b

    def login(self):
        """Logs into the website."""
        try:
            self.driver.get(WEB_CONFIG["urls"]["base"])
            logger.info(f"Navigated to base URL: {WEB_CONFIG['urls']['base']}")

            username_field = self.wait.until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Username/email']")))
            username_field.send_keys(WEB_CONFIG["credentials"]["username"])
            
            password_field = self.driver.find_element(By.XPATH, "//input[@placeholder='Password']")
            password_field.send_keys(WEB_CONFIG["credentials"]["password"])
            password_field.send_keys(Keys.RETURN)
            
            # Add a wait condition for successful login, e.g., an element on the dashboard
            # For now, a short sleep, but explicit wait is better.
            time.sleep(5) # Replace with explicit wait for a post-login element
            logger.info("Login attempt submitted.")
            # Example: self.wait.until(EC.presence_of_element_located((By.ID, "user-dashboard-element")))
            return True
        except TimeoutException:
            logger.error("Timeout during login process.")
            return False
        except Exception as e:
            logger.error(f"An error occurred during login: {e}")
            return False

    def _explore_frame_recursive(self, current_depth=0):
        """Recursively searches for and switches to the target iframe."""
        iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
        logger.debug(f"{'  ' * current_depth}Found {len(iframes)} iframes at depth {current_depth}")

        for index, iframe_element in enumerate(iframes):
            try:
                src = iframe_element.get_attribute('src')
                logger.debug(f"{'  ' * current_depth}Checking iframe {index} with src: {src}")
                
                self.driver.switch_to.frame(iframe_element)
                
                if WEB_CONFIG["urls"]["iframe_pattern"] in str(self.driver.current_url) or \
                   WEB_CONFIG["urls"]["iframe_pattern"] in str(src): # Check current_url after switch
                    logger.info(f"{'  ' * current_depth}Found and switched to target iframe: {src}")
                    return True
                
                if self._explore_frame_recursive(current_depth + 1):
                    return True # Target found in nested iframe
                
                self.driver.switch_to.parent_frame() # Back to parent
            except NoSuchElementException:
                logger.warning(f"{'  ' * current_depth}Iframe {index} became stale or was not found during switch.")
                # If an error occurs, try to switch back to parent and continue with the next iframe
                try:
                    self.driver.switch_to.parent_frame()
                except WebDriverException: # Could be already at default content
                    self.driver.switch_to.default_content()
            except Exception as e:
                logger.error(f"{'  ' * current_depth}Error exploring iframe {index} (src: {src}): {e}")
                try:
                    self.driver.switch_to.parent_frame()
                except WebDriverException:
                    self.driver.switch_to.default_content()
        return False

    def switch_to_game_iframe(self):
        """Switches the WebDriver context to the game iframe."""
        try:
            self.driver.switch_to.default_content() # Ensure starting from main document
            if self._explore_frame_recursive():
                logger.info("Successfully switched to target game iframe.")
                # Wait for an element within the iframe to ensure it's loaded
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".value--dd5c7"))) # Example element
                return True
            else:
                logger.error("Target game iframe not found after recursive search.")
                return False
        except Exception as e:
            logger.error(f"Exception while trying to switch to game iframe: {e}")
            return False

    def extract_roulette_numbers_js(self):
        """Executes JavaScript to extract numbers from the roulette game."""
        js_code = """
        try {
            const numberElements = document.querySelectorAll('.value--dd5c7'); // Selector for numbers
            if (numberElements.length === 0) return null; // No numbers found
            const numbers = Array.from(numberElements).map(el => parseInt(el.textContent?.trim() || '-1', 10));
            // Filter out any -1 which might result from bad parsing or empty elements
            const validNumbers = numbers.filter(n => n !== -1 && n >= 0 && n <= 36);
            if (validNumbers.length === 0) return null;
            return { numbers: validNumbers }; // Return only numbers for now
        } catch (e) {
            return { error: e.toString() };
        }
        """
        try:
            result = self.driver.execute_script(js_code)
            if result and "error" in result:
                logger.error(f"JavaScript error during number extraction: {result['error']}")
                return None
            if result and result.get("numbers"):
                logger.debug(f"Extracted numbers via JS: {result['numbers']}")
                return result["numbers"]
            logger.warning("JavaScript execution did not return numbers or result was null.")
            return None
        except Exception as e:
            logger.error(f"Error executing JavaScript for number extraction: {e}")
            return None
            
    def collect_initial_data(self, num_rounds_to_collect=PRED_CONFIG["window_size"] + 20):
        """Collects an initial set of numbers before starting predictions."""
        logger.info(f"Starting initial data collection for {num_rounds_to_collect} rounds.")
        collected_numbers_history = []
        last_seen_numbers = []

        while len(collected_numbers_history) < num_rounds_to_collect:
            current_numbers = self.extract_roulette_numbers_js()
            if current_numbers:
                # Check if the most recent number is new
                if not last_seen_numbers or current_numbers[0] != last_seen_numbers[0]:
                    if len(current_numbers) > len(last_seen_numbers) or current_numbers[0] != last_seen_numbers[0]:
                        new_number = current_numbers[0] # Newest number is at the start
                        collected_numbers_history.insert(0, new_number) # Add to beginning to maintain order
                        logger.info(f"Collected new number: {new_number}. Total collected: {len(collected_numbers_history)}/{num_rounds_to_collect}")
                        last_seen_numbers = list(current_numbers) # Update last_seen_numbers
                else:
                    logger.debug("No new number detected yet.")
            else:
                logger.warning("Failed to extract numbers during initial collection.")
            
            time.sleep(10) # Wait for the next spin or numbers to update
            if keyboard.is_pressed('esc'): # Allow manual interruption
                logger.info("Initial data collection interrupted by user.")
                break
        
        logger.info(f"Initial data collection complete. Collected: {collected_numbers_history}")
        return collected_numbers_history # Returns in chronological order (oldest to newest)


    def run_roulette_bot(self):
        """Main loop for the roulette bot."""
        if not self.login():
            logger.error("Login failed. Exiting bot.")
            return

        if not self.switch_to_game_iframe():
            logger.error("Could not switch to game iframe. Exiting bot.")
            return

        logger.info("Successfully logged in and switched to iframe. Starting data collection and prediction loop.")
        
        # Initial data collection phase (e.g., collect window_size + a few more numbers)
        # The predictor needs at least `window_size` numbers to make a prediction.
        # We collect them in chronological order (oldest first).
        historical_numbers = self.collect_initial_data()
        if len(historical_numbers) < PRED_CONFIG["window_size"]:
            logger.error(f"Not enough initial data collected ({len(historical_numbers)}). Need at least {PRED_CONFIG['window_size']}. Exiting.")
            return

        last_processed_numbers_snapshot = list(historical_numbers) # Keep a snapshot of numbers used for last prediction

        try:
            while True:
                if keyboard.is_pressed('esc'): # Allow manual interruption
                    logger.info("Roulette bot run interrupted by user.")
                    break

                current_on_screen_numbers = self.extract_roulette_numbers_js()

                if not current_on_screen_numbers:
                    logger.warning("Could not extract numbers from screen. Retrying...")
                    time.sleep(10) # Wait before retrying
                    # Potentially add logic to re-check iframe or page state
                    if not self.switch_to_game_iframe(): # Try to re-acquire iframe
                         logger.error("Lost iframe, cannot continue.")
                         break
                    continue

                # The `current_on_screen_numbers` are usually in reverse chronological order (newest first)
                # We need to see if a new number has appeared compared to our `historical_numbers`
                
                # If current_on_screen_numbers[0] is different from historical_numbers[-1] (the newest we know)
                # it means a new spin has occurred.
                if not historical_numbers or current_on_screen_numbers[0] != historical_numbers[-1]:
                    new_outcome = current_on_screen_numbers[0]
                    logger.info(f"New spin detected! Outcome: {new_outcome}")
                    
                    # The input for prediction should be the `window_size` numbers *before* this new_outcome
                    # So, historical_numbers should contain these.
                    if len(historical_numbers) >= PRED_CONFIG["window_size"]:
                        input_sequence_for_prediction = historical_numbers[-PRED_CONFIG["window_size"]:]
                        
                        logger.info(f"Predicting based on sequence: {input_sequence_for_prediction}")
                        
                        # Determine bet amount using Fibonacci strategy
                        self.current_bet_amount = self._get_fibonacci_bet(self.losing_streak)
                        logger.info(f"Losing streak: {self.losing_streak}. Betting: {self.current_bet_amount} units.")

                        # Record round, which also calls predictor.train and predictor.predict_section
                        # The predictor's internal balance will be updated.
                        # The `record_round_and_train` method in WheelPredictor handles prediction,
                        # training data addition, balance update, and model training trigger.
                        predicted_section = self.predictor.record_round_and_train(
                            input_numbers=list(input_sequence_for_prediction), # Ensure it's a list copy
                            outcome_number=new_outcome,
                            bet_amount=self.current_bet_amount # Pass the actual bet amount
                        )
                        logger.info(f"Outcome: {new_outcome}, Predicted Section: {predicted_section}, Current Balance (predictor): {self.predictor.balance}")

                        # Update losing streak based on prediction outcome
                        if new_outcome in predicted_section:
                            logger.info("WIN! Resetting losing streak.")
                            self.losing_streak = 0
                        else:
                            self.losing_streak += 1
                            logger.info(f"LOSS. Losing streak increased to: {self.losing_streak}")
                            if self.losing_streak >= WEB_CONFIG.get("fibonacci_max_streak", 10):
                                logger.warning(f"Max losing streak ({self.losing_streak}) reached. Resetting streak and bet.")
                                self.losing_streak = 0 # Reset streak or implement other strategy

                    else:
                        logger.info(f"Not enough historical data ({len(historical_numbers)}) to make a prediction yet. Need {PRED_CONFIG['window_size']}.")

                    # Add the new outcome to our historical_numbers
                    historical_numbers.append(new_outcome)
                    # Keep historical_numbers from growing indefinitely if not managed by predictor's deque
                    if len(historical_numbers) > PRED_CONFIG["training_deque_maxlen"] + PRED_CONFIG["window_size"]: # Keep a bit more than deque
                        historical_numbers.pop(0) # Remove oldest

                else:
                    logger.debug("No new spin detected. Numbers on screen match last known historical number.")

                time.sleep(15) # Wait for the next spin cycle (adjust as needed)

        except KeyboardInterrupt:
            logger.info("Roulette bot process interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main bot loop: {e}", exc_info=True)
        finally:
            self.predictor.save_state() # Ensure state is saved on exit
            self.quit_driver()

    def quit_driver(self):
        """Closes the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed.")
            except Exception as e:
                logger.error(f"Error quitting WebDriver: {e}")

if __name__ == '__main__':
    logger.info("Starting Roulette Bot directly from web_scraper.py")
    
    # Initialize the predictor
    try:
        roulette_predictor = WheelPredictor()
    except Exception as e:
        logger.error(f"Failed to initialize WheelPredictor: {e}", exc_info=True)
        exit(1)

    # Initialize and run the scraper/bot
    try:
        bot = RouletteScraper(predictor_instance=roulette_predictor)
        bot.run_roulette_bot()
    except WebDriverException as e:
        logger.error(f"WebDriver related error during bot execution: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unhandled exception during bot execution: {e}", exc_info=True)
    finally:
        logger.info("Roulette Bot finished or encountered an error.")
