import logging
from config import LOGGING_CONFIG # For setting up logging
from wheel_predictor import WheelPredictor
from web_scraper import RouletteScraper
from selenium.common.exceptions import WebDriverException

# Setup logging as early as possible
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"]
)
logger = logging.getLogger(__name__)

def run_application():
    """
    Initializes and runs the Roulette Predictor and Scraper application.
    """
    logger.info("Starting Roulette Prediction Application...")

    try:
        # Initialize the predictor
        logger.info("Initializing Wheel Predictor...")
        predictor = WheelPredictor()
        logger.info("Wheel Predictor initialized successfully.")
    except Exception as e:
        logger.error(f"Fatal Error: Failed to initialize WheelPredictor: {e}", exc_info=True)
        return # Exit if predictor fails

    try:
        # Initialize the scraper with the predictor instance
        logger.info("Initializing Roulette Scraper...")
        scraper = RouletteScraper(predictor_instance=predictor)
        logger.info("Roulette Scraper initialized successfully.")

        # Run the main bot logic
        logger.info("Starting the roulette bot...")
        scraper.run_roulette_bot()

    except WebDriverException as e:
        logger.error(f"Fatal Error: WebDriver issue during bot execution: {e}", exc_info=True)
        if 'scraper' in locals() and scraper.driver:
             logger.info("Attempting to quit WebDriver...")
             scraper.quit_driver()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
        if 'scraper' in locals() and scraper.driver:
             logger.info("Attempting to quit WebDriver due to interruption...")
             scraper.quit_driver()
    except Exception as e:
        logger.error(f"Fatal Error: An unhandled exception occurred in the application: {e}", exc_info=True)
        if 'scraper' in locals() and scraper.driver:
             logger.info("Attempting to quit WebDriver due to an unhandled exception...")
             scraper.quit_driver()
    finally:
        logger.info("Roulette Prediction Application finished.")

if __name__ == '__main__':
    run_application()
