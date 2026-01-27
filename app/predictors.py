import os
import numpy as np
from tensorflow import keras
import traceback


class RegressionPredictor:
    _model = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_path = "app/models/regression.h5"
            if os.path.exists(model_path):
                cls._model = keras.models.load_model(model_path)
                print(f"Regression model loaded from {model_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    "Please train the model first by running: python regression.py"
                )
        return cls._instance

    @classmethod
    def predict(cls, input_data):
        if cls._model is None:
            raise RuntimeError("Model not initialized")

        x = np.array(input_data).astype("float32") / 255.0

        if x.size == 784:
            x = x.reshape(28, 28)

        if x.shape != (28, 28):
            raise ValueError(f"Expected (28,28), got {x.shape}")

        x = x.reshape(1, 28, 28)

        predictions = cls._model.predict(x, verbose=0)
        return predictions.flatten().tolist()



class CNNPredictor:
    _model = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_path = "app/models/convolutional.h5"
            if os.path.exists(model_path):
                cls._model = keras.models.load_model(model_path)
                print(f"CNN model loaded from {model_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    "Please train the model first by running: python convolutional.py"
                )
        return cls._instance

    @classmethod
    def predict(cls, input_data):
        if cls._model is None:
            raise RuntimeError("Model not initialized. Use 'CNNPredictor()' first.")

        try:
            x = np.array(input_data).astype("float32")

            if x.shape == (784,):
                x = x.reshape(1, 28, 28, 1)
            elif x.shape == (28, 28):
                x = x.reshape(1, 28, 28, 1)
            elif x.shape == (1, 28, 28):
                x = x.reshape(1, 28, 28, 1)
            elif x.shape == (1, 28, 28, 1):
                pass
            else:
                raise ValueError(f"Invalid CNN input shape: {x.shape}")

            x = x / 255.0

            predictions = cls._model.predict(x, verbose=0)
            return predictions.flatten().tolist()

        except Exception:
            print("Error inside CNNPredictor.predict:")
            traceback.print_exc()
            raise
