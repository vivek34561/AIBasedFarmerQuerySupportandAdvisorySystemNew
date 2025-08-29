import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("model", "model8.keras")  # <- your saved model
        self.class_indices_path = os.path.join("model", "class_indices.json")  # <- class mappings

        # Load trained model
        self.model = load_model(self.model_path)

        # Load class indices
        with open(self.class_indices_path, "r") as f:
            self.class_indices = json.load(f)

        # Reverse mapping {index: class_name}
        self.class_labels = {v: k for k, v in self.class_indices.items()}

    def predict(self):
        # Preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = self.model.predict(img_array)
        predicted_class_index = np.argmax(preds, axis=1)[0]
        predicted_class = self.class_labels[predicted_class_index]

        # Return prediction + probabilities
        return [{
            "image": predicted_class,
            "probabilities": preds[0].tolist()
        }]
