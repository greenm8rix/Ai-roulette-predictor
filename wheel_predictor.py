import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
import logging

# Custom modules
from config import CONFIG, WHEEL_ORDER, NUM_TO_POS
from model_utils import Attention

logger = logging.getLogger(__name__)

# -----------------------------
# Wheel Predictor Class
# -----------------------------
class WheelPredictor:
    """
    Predicts roulette wheel outcomes based on historical data using an LSTM model
    with an attention mechanism.
    """
    def __init__(self):
        self.window_size = CONFIG["window_size"]
        self.prediction_size = CONFIG["prediction_size"]
        self.balance = CONFIG["balance"]
        self.training_data = collections.deque(maxlen=CONFIG["training_deque_maxlen"])
        self.model = self._build_or_load_model()
        self._load_state()

    def _build_or_load_model(self):
        """
        Loads an existing Keras model or builds a new one if not found or corrupted.
        """
        try:
            model = load_model(
                CONFIG["model_file"],
                custom_objects={"Attention": Attention}
            )
            logger.info(f"Loaded existing model from {CONFIG['model_file']}")
            # Re-compile is often good practice after loading, especially if optimizer state needs reset
            model.compile(
                optimizer=tf.keras.optimizers.Adam(CONFIG["learning_rate"]),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        except Exception as e:
            logger.warning(f"Creating new model; error loading existing model from {CONFIG['model_file']}: {e}")
            inputs = Input(shape=(self.window_size, 2)) # 2 for sin/cos cyclical features

            if CONFIG["bidirectional"]:
                x = Bidirectional(
                    LSTM(CONFIG["lstm_units1"], return_sequences=True, kernel_regularizer=regularizers.l2(CONFIG["l2_reg"]))
                )(inputs)
            else:
                x = LSTM(CONFIG["lstm_units1"], return_sequences=True, kernel_regularizer=regularizers.l2(CONFIG["l2_reg"]))(inputs)
            x = Dropout(CONFIG["dropout_rate1"])(x)

            # Second LSTM layer (consider if return_sequences should be True for Attention)
            x = LSTM(CONFIG["lstm_units2"], return_sequences=True, kernel_regularizer=regularizers.l2(CONFIG["l2_reg"]))(x)
            x = Dropout(CONFIG["dropout_rate2"])(x)

            context = Attention()(x)
            x = Dense(
                CONFIG["dense_units"],
                activation="relu",
                kernel_regularizer=regularizers.l2(CONFIG["l2_reg"])
            )(context)
            outputs = Dense(len(WHEEL_ORDER), activation="softmax")(x) # len(WHEEL_ORDER) is 37

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(CONFIG["learning_rate"]),
                loss="categorical_crossentropy", # Suitable for multi-class classification
                metrics=["accuracy"]
            )
            logger.info("Created and compiled a new model.")
        return model

    def _load_state(self):
        """Loads predictor state (training data, balance) from a pickle file."""
        try:
            with open(CONFIG["state_file"], "rb") as f:
                state = pickle.load(f)
                self.training_data = collections.deque(
                    state.get("training_data", []), maxlen=CONFIG["training_deque_maxlen"]
                )
                self.balance = state.get("balance", CONFIG["balance"])
                logger.info(f"Loaded predictor state from {CONFIG['state_file']}. Training data size: {len(self.training_data)}, Balance: {self.balance}")
        except FileNotFoundError:
            logger.info(f"No previous state file found at {CONFIG['state_file']}. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading state from {CONFIG['state_file']}: {e}. Starting fresh.")


    def save_state(self):
        """Saves predictor state (training data, balance) to a pickle file."""
        state = {
            "training_data": list(self.training_data),
            "balance": self.balance
        }
        try:
            with open(CONFIG["state_file"], "wb") as f:
                pickle.dump(state, f)
            logger.info(f"Predictor state saved to {CONFIG['state_file']}.")
        except Exception as e:
            logger.error(f"Error saving state to {CONFIG['state_file']}: {e}")

    @staticmethod
    def _convert_numbers_to_positions(numbers):
        """Converts roulette numbers to their positional indices on the wheel."""
        return [NUM_TO_POS[num] for num in numbers]

    def _validate_input_numbers(self, input_numbers):
        """Validates if the input numbers list is suitable for prediction."""
        if not isinstance(input_numbers, list):
            logger.warning("Invalid input_numbers type. Expected list.")
            return False
        if len(input_numbers) < self.window_size:
            logger.warning(f"Not enough input numbers. Expected at least {self.window_size}, got {len(input_numbers)}.")
            return False
        if not all(isinstance(n, int) and 0 <= n < len(WHEEL_ORDER) for n in input_numbers):
            logger.warning(f"Invalid numbers in input_numbers. All must be integers between 0 and {len(WHEEL_ORDER)-1}.")
            return False
        return True

    def _rotate_sample(self, input_positions, outcome_position):
        """Applies rotational data augmentation to a single sample."""
        augmented_samples = []
        num_rotations = CONFIG.get("augmentation_rotations", 0)
        for _ in range(num_rotations):
            # Ensure rotation is non-zero to create a different sample
            rot = np.random.randint(1, len(WHEEL_ORDER)) # Max rotation is 36 for a 37-number wheel
            rotated_input = [(pos + rot) % len(WHEEL_ORDER) for pos in input_positions]
            rotated_outcome = (outcome_position + rot) % len(WHEEL_ORDER)
            augmented_samples.append((rotated_input, rotated_outcome))
        return augmented_samples

    def _preprocess_input_for_model(self, positions):
        """Converts a list of positions into cyclical features for the model."""
        cyclical_features = np.zeros((1, self.window_size, 2)) # Batch size of 1
        for i, pos in enumerate(positions):
            angle = 2 * np.pi * pos / len(WHEEL_ORDER)
            cyclical_features[0, i, 0] = np.sin(angle)
            cyclical_features[0, i, 1] = np.cos(angle)
        return cyclical_features

    def _prepare_training_batch(self):
        """Prepares X and y batches for training, including cyclical conversion and label smoothing."""
        X_pos, y_pos = zip(*self.training_data)
        X_pos = np.array(X_pos, dtype=np.float32)
        y_pos = np.array(y_pos, dtype=np.int32)

        # Convert input positions to cyclical features
        X_cyclical = np.zeros((X_pos.shape[0], self.window_size, 2))
        for i in range(X_pos.shape[0]):
            for j in range(self.window_size):
                pos = X_pos[i, j]
                angle = 2 * np.pi * pos / len(WHEEL_ORDER)
                X_cyclical[i, j, 0] = np.sin(angle)
                X_cyclical[i, j, 1] = np.cos(angle)

        # Gaussian-smoothed labels for y
        sigma = CONFIG.get("label_smoothing_sigma", 3.0) # Default sigma if not in config
        y_smoothed = np.zeros((len(y_pos), len(WHEEL_ORDER)))
        for i, pos_idx in enumerate(y_pos):
            distances = np.minimum(np.abs(np.arange(len(WHEEL_ORDER)) - pos_idx),
                                   len(WHEEL_ORDER) - np.abs(np.arange(len(WHEEL_ORDER)) - pos_idx))
            y_smoothed[i] = np.exp(-(distances**2) / (2 * sigma**2))
            y_smoothed[i] /= np.sum(y_smoothed[i]) # Normalize to sum to 1

        return X_cyclical, y_smoothed


    def train_model_on_batch(self):
        """Trains the model on the current accumulated training_data."""
        if len(self.training_data) < CONFIG["min_training_samples"]:
            logger.info(f"Not enough training samples ({len(self.training_data)}) for batch training. Min required: {CONFIG['min_training_samples']}.")
            return

        X_cyclical, y_smoothed = self._prepare_training_batch()

        X_train, X_val, y_train, y_val = train_test_split(
            X_cyclical, y_smoothed, test_size=0.2, shuffle=True, random_state=42 # Added random_state for reproducibility
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=CONFIG.get("early_stopping_patience", 50), restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=CONFIG.get("reduce_lr_patience", 25), factor=0.2, verbose=1)
        ]

        logger.info(f"Starting batch training with {len(X_train)} samples...")
        history = self.model.fit(
            X_train, y_train,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks
        )
        logger.info("Batch training complete.")
        self.model.save(CONFIG["model_file"]) # Save in modern Keras format
        logger.info(f"Model saved to {CONFIG['model_file']}")
        return history


    def add_training_sample(self, input_numbers, outcome_number):
        """
        Adds a new sample to the training data and applies augmentation.
        Assumes input_numbers are raw roulette numbers.
        """
        if not self._validate_input_numbers(input_numbers) or \
           not (isinstance(outcome_number, int) and 0 <= outcome_number < len(WHEEL_ORDER)):
            logger.error(f"Invalid data for training sample: inputs={input_numbers}, outcome={outcome_number}")
            return False

        recent_numbers = input_numbers[-self.window_size:]
        input_positions = self._convert_numbers_to_positions(recent_numbers)
        outcome_position = NUM_TO_POS[outcome_number]

        self.training_data.append((input_positions, outcome_position))
        logger.debug(f"Added original sample. Training data size: {len(self.training_data)}")

        if CONFIG.get("data_augmentation", False):
            augmented_samples = self._rotate_sample(input_positions, outcome_position)
            for aug_input_pos, aug_outcome_pos in augmented_samples:
                self.training_data.append((aug_input_pos, aug_outcome_pos))
            logger.debug(f"Added {len(augmented_samples)} augmented samples. Training data size: {len(self.training_data)}")
        return True


    def predict_roulette_section(self, input_numbers, ensemble_runs=5):
        """
        Predicts a section of the roulette wheel.
        Input_numbers should be a list of the most recent raw roulette numbers.
        """
        if not self._validate_input_numbers(input_numbers):
            logger.warning("Prediction skipped due to invalid input numbers.")
            # Return a default or empty prediction
            return [WHEEL_ORDER[i % len(WHEEL_ORDER)] for i in range(self.prediction_size)]


        recent_numbers = input_numbers[-self.window_size:]
        input_positions = self._convert_numbers_to_positions(recent_numbers)
        model_input = self._preprocess_input_for_model(input_positions)

        all_probs = []
        # MC Dropout for uncertainty estimation if ensemble_runs > 1
        use_mc_dropout = ensemble_runs > 1
        for _ in range(ensemble_runs):
            # training=True enables dropout during inference for MC Dropout
            probs = self.model(model_input, training=use_mc_dropout).numpy()[0]
            all_probs.append(probs)

        # Average probabilities if MC Dropout was used
        avg_probs = np.mean(all_probs, axis=0)

        # Determine the best starting index for the prediction_size section
        # This sums probabilities for contiguous sections of prediction_size on the wheel
        cyclic_sums = [
            sum(avg_probs[(i + j) % len(WHEEL_ORDER)] for j in range(self.prediction_size))
            for i in range(len(WHEEL_ORDER)) # Iterate through all possible start positions
        ]
        best_start_pos_index = int(np.argmax(cyclic_sums)) # Index in the WHEEL_ORDER

        # Construct the predicted section using WHEEL_ORDER
        predicted_section_numbers = [
            WHEEL_ORDER[(best_start_pos_index + i) % len(WHEEL_ORDER)]
            for i in range(self.prediction_size)
        ]
        logger.info(f"Predicted section: {predicted_section_numbers} based on input: ...{recent_numbers[-5:]}")
        return predicted_section_numbers


    def record_round_and_train(self, input_numbers, outcome_number, bet_amount):
        """
        Records the outcome of a round, updates balance, adds data for training,
        and potentially triggers model training.
        """
        if not self.add_training_sample(input_numbers, outcome_number):
            return # Invalid data, skip processing

        predicted_section = self.predict_roulette_section(input_numbers, ensemble_runs=CONFIG.get("mc_dropout_runs", 5))

        # Update balance based on whether the outcome was in the predicted section
        # This is a simplified betting simulation logic
        if outcome_number in predicted_section:
            # Simplified win: assume a fixed payout for being in the section
            # This needs to be more specific based on actual betting rules (e.g., betting on a section)
            # For now, let's assume a conceptual win if the number falls in the predicted zone.
            # Payout logic needs to be defined. If betting on a section of 10 numbers,
            # a win on one of those numbers would typically pay less than 35:1.
            # Let's assume a conceptual "correct prediction" payout for simplicity.
            # Example: win 2 times the bet if correct.
            winnings = bet_amount * 2 # Placeholder for actual payout calculation
            self.balance += winnings - bet_amount # Net gain
            logger.info(f"✅ Correct prediction! Outcome {outcome_number} in {predicted_section}. Bet: {bet_amount}, Won: {winnings}. New balance: {self.balance}")
        else:
            self.balance -= bet_amount
            logger.info(f"❌ Incorrect prediction. Outcome {outcome_number} not in {predicted_section}. Bet: {bet_amount}. New balance: {self.balance}")

        # Periodically train the model (e.g., every N rounds or if data size threshold met)
        # The original code trained every 100 appends to training_data.
        # Let's make this configurable or based on a number of new samples.
        if len(self.training_data) > 0 and len(self.training_data) % CONFIG.get("train_every_n_samples", 100) == 0:
            self.train_model_on_batch()

        self.save_state() # Save state after each round
        return predicted_section

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    predictor = WheelPredictor()

    # Simulate some data
    test_input_sequence = [WHEEL_ORDER[i % len(WHEEL_ORDER)] for i in range(20)] # A sequence of 20 numbers
    
    # Test adding a training sample
    print("\n--- Testing Training Sample Addition ---")
    sample_input = test_input_sequence[:CONFIG["window_size"]]
    sample_outcome = test_input_sequence[CONFIG["window_size"]]
    print(f"Input for training: {sample_input}, Outcome: {sample_outcome}")
    predictor.add_training_sample(sample_input, sample_outcome)
    print(f"Training data size: {len(predictor.training_data)}")

    # Test prediction
    print("\n--- Testing Prediction ---")
    # Use a sequence of numbers that meets the window_size
    current_numbers = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13] # Exactly window_size
    if len(current_numbers) >= CONFIG["window_size"]:
        predicted_nums = predictor.predict_roulette_section(current_numbers)
        print(f"Input for prediction: {current_numbers}")
        print(f"Predicted numbers: {predicted_nums}")
    else:
        print(f"Not enough numbers for prediction. Need {CONFIG['window_size']}, got {len(current_numbers)}")

    # Test training (if enough data is simulated or loaded)
    print("\n--- Testing Model Training ---")
    # To actually train, you'd need to add more samples or have a pre-existing state
    # For this example, we'll just call it to see if it runs without error
    # predictor.train_model_on_batch() # This will likely print "Not enough training samples"

    # Simulate a few rounds for record_round_and_train
    print("\n--- Testing Record Round and Train ---")
    for i in range(CONFIG["window_size"], len(test_input_sequence) -1):
        input_seq = test_input_sequence[i - CONFIG["window_size"] : i]
        outcome_val = test_input_sequence[i]
        bet = 10
        print(f"\nRound: Input: {input_seq}, Outcome: {outcome_val}, Bet: {bet}")
        predictor.record_round_and_train(input_seq, outcome_val, bet)
        print(f"Balance after round: {predictor.balance}")
        if (i+1) % 5 == 0 and len(predictor.training_data) >= CONFIG["min_training_samples"]: # Train every 5 rounds if enough data
             print("Triggering batch training...")
             predictor.train_model_on_batch()


    print(f"\nFinal balance: {predictor.balance}")
    print(f"Final training data size: {len(predictor.training_data)}")
    predictor.save_state()
    # predictor.model.summary() # Optional: print model summary
