from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras import metrics
import keras as K

import tensorflow as tf

from typing import List
import pathlib
import os


class Model:
    def __init__(self):
        print("Versão do TensorFlow:", tf.__version__)
        print("Versão do Keras:", K.__version__)
        print(
            f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}"
        )
        tf.keras.backend.clear_session()
        # Init variables
        self.model: Sequential = None
        self.history: K.callbacks.History = None
        self.predicted: List[int] = None

        self.train_path: os.path = None
        self.test_path: os.path = None
        self.test_path: os.path = None

        self.training_set: K.Image.DirectoryIterator = None
        self.validation_set: K.Image.DirectoryIterator = None
        self.testing_set: K.Image.DirectoryIterator = None

    def set_images_path(self, colab: bool = False):
        if colab:
            from google.colab import drive

            drive.mount("/content/drive")
            current_path = "/content/drive"
        else:
            current_path = str(pathlib.Path().absolute())
        images_root_path = os.path.join(current_path, "data")
        self.train_path = os.path.join(images_root_path, "train")
        self.test_path = os.path.join(images_root_path, "test")
        self.validation_path = os.path.join(images_root_path, "val")

        print(f"Train PATH: {self.train_path}")
        print(f"Test PATH: {self.test_path}")
        print(f"Vaidation PATH: {self.validation_path}")

    def set_data(
        self,
        batch_size: int = 32,
        image_height: int = 64,
        image_width: int = 64,
    ) -> Sequential:
        # TRAIN DATASET
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.training_set = train_datagen.flow_from_directory(
            self.train_path,
            seed=42,
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode="binary",
            subset="training",
        )

        # VALIDATION DATASET
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.validation_set = validation_datagen.flow_from_directory(
            self.validation_path,
            seed=42,
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode="binary",
            subset="validation",
        )

        # TEST DATASET
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.testing_set = test_datagen.flow_from_directory(
            self.test_path,
            seed=42,
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode="binary",
        )

    def set_model(self):
        self.model = Sequential()

        self.model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(6, 6),
                input_shape=(64, 64, 3),
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(2, 2),
                kernel_initializer="he_uniform",
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(2, 2),
                kernel_initializer="he_uniform",
                activation="relu",
            )
        )

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dense(units=1, activation="sigmoid"))

        # Compilando a rede
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=[
                metrics.FalseNegatives(name="fn"),
                metrics.BinaryAccuracy(name="accuracy"),
            ],
        )

    def train_and_predict(self):
        self.train_model()
        return self.predict_model()

    def train_model(self, persist_model: bool = True):
        self.history = self.model.fit(
            self.training_set,
            steps_per_epoch=len(self.training_set),
            epochs=5,
            validation_data=self.validation_set,
            validation_steps=len(self.validation_set),
        )

        if persist_model:
            self.model.save("saved_model")

        self.model.summary()
        return self.model

    def predict_model(self) -> List[int]:
        self.model.evaluate(self.testing_set, batch_size=10)
        self.predicted = (self.model.predict(self.testing_set) > 0.5).astype("int32")
        self.predicted = [pred[0] for pred in predicted]
        return self.predicted

    # def show_metrics(self):


if __name__ == "__main__":
    model = Model()
    model.set_images_path()
    model.set_data()
    model.set_model()
    predicted = model.train_and_predict()
    print(predicted)
