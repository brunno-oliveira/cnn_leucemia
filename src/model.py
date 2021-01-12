import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras import metrics
import keras as K

from sklearn.metrics import accuracy_score, recall_score

import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import numpy as np
import pathlib
import random
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


class Model:
    def __init__(self, env: str = "local"):
        if env == "local":
            self.data_path = os.path.join(str(Path().absolute()), "data")
        elif env == "colab":
            self.data_path = os.path.join(
                str(Path().absolute()), "cnn_leucemia", "data"
            )
        else:
            raise Exception(f"ENV {env} invalid!")
        print(f"Versão do TensorFlow: {tf.__version__}")
        print(f"Versão do Keras: {K.__version__}")
        print(
            f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}"
        )

        tf.keras.backend.clear_session()
        # Init variables
        self.model: Sequential = None
        self.history: K.callbacks.History = None
        self.predicted: List[int] = None
        self.predicted_proba: List[float] = None

        self.train_path: os.path = None
        self.test_path: os.path = None
        self.test_path: os.path = None

        self.training_set: K.Image.DirectoryIterator = None
        self.validation_set: K.Image.DirectoryIterator = None
        self.testing_set: K.Image.DirectoryIterator = None

    def set_images_path(self):
        print("SET IMAGES PATH")
        images_root_path = os.path.join(self.data_path, "data")
        self.train_path = os.path.join(self.data_path, "train")
        self.test_path = os.path.join(self.data_path, "test")
        self.validation_path = os.path.join(self.data_path, "val")

        print(f"Train PATH: {self.train_path}")
        print(f"Test PATH: {self.test_path}")
        print(f"Vaidation PATH: {self.validation_path}")

    def set_data(self, batch_size: int = 32) -> Sequential:
        print("SET DATA")
        image_height: int = 64
        image_width: int = 64
        # TRAIN DATASET

        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.training_set = train_datagen.flow_from_directory(
            self.train_path,
            seed=SEED,
            color_mode="grayscale",
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode="binary",
        )

        # VALIDATION DATASET
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.validation_set = validation_datagen.flow_from_directory(
            self.validation_path,
            seed=SEED,
            color_mode="grayscale",
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode="binary",
        )

        # TEST DATASET
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.testing_set = test_datagen.flow_from_directory(
            self.test_path,
            seed=SEED,
            color_mode="grayscale",
            target_size=(image_height, image_width),
            batch_size=batch_size,
            class_mode="binary",
        )

    def set_model(self):
        self.model = Sequential()

        self.model.add(
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(64, 64, 1),
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(
                filters=1024,
                kernel_size=(3, 3),
                activation="relu",
            )
        )

        self.model.add(
            layers.Conv2D(
                filters=1024,
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.6))
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.6))
        self.model.add(layers.Dense(units=1, activation="sigmoid"))

        # Compilando a rede
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="mse",
            metrics=[
                metrics.BinaryAccuracy(name="accuracy"),
                metrics.Recall(name="recall"),
            ],
        )

        self.model.summary()

    def train_and_predict(self, persist_model: bool = True, epochs: int = 5):
        print("TRAIN PREDICT MODEL")
        self.train_model(persist_model, epochs)
        self.predict_model()

    def train_model(self, persist_model: bool = True, epochs: int = 5):
        print("TRAIN MODEL")
        self.history = self.model.fit(
            self.training_set,
            steps_per_epoch=len(self.training_set),
            epochs=epochs,
            validation_data=self.validation_set,
            validation_steps=len(self.validation_set),
        ).history

        if persist_model:
            self.model.save("saved_model")
        return self.model

    def predict_model(self) -> List[int]:
        print("PREDICT MODEL")
        self.model.evaluate(self.testing_set, batch_size=10)
        self.predicted_proba = (self.model.predict(self.testing_set) > 0.5).astype(
            "int32"
        )
        self.predicted = [pred[0] for pred in self.predicted_proba]
        return self.predicted

    def plot_train_metrics(self):
        print("PLOT TRAIN METRICS")

        plt.plot(self.history["loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim([0.01, 1])
        plt.legend(loc="upper right")
        plt.show()

        plt.plot(self.history["accuracy"], label="Training Accuracy")
        plt.plot(self.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.5, 1])
        plt.legend(loc="lower right")
        plt.show()

        plt.plot(self.history["recall"], label="Training Recall")
        plt.plot(self.history["val_recall"], label="Validation Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.ylim([0.5, 1])
        plt.legend(loc="lower right")
        plt.show()

    def plot_test_metrics(self):
        print("PLOT TEST METRICS")
        acc = round(
            accuracy_score(self.testing_set.classes, np.array(self.predicted)),
            3,
        )
        print(f"Accuracy: {acc}")

        recall = round(
            recall_score(self.testing_set.classes, np.array(self.predicted)), 3
        )
        print(f"Recall: {recall}")

        tf.math.confusion_matrix(self.testing_set.classes, self.predicted)


if __name__ == "__main__":
    model = Model()
    model.set_images_path()
    model.set_data()
    model.set_model()
    model.train_and_predict()
