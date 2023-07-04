import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt

EPOCHS = 100
BATCH_SIZE = 16
MODEL_EPOCHS = [3, 6, 9]
NAMES = ["model1", "model2", "model3", "ultimate_model"]
DATA_FILE_NAME = "./train.csv"

def load_data_from_csv(file_path):
    csv_data = pd.read_csv(file_path)
    data = np.array(csv_data)
    return data

def preprocess_data(csv_data):
    labels = np.array([row[0] for row in csv_data])
    vectors = np.array([(np.array(row[1:]) / 255).reshape(-1, 28) for row in csv_data]).reshape(-1, 28, 28, 1)
    return vectors, labels

def get_accuracy(labels, predictions):
    return len([1 for i in range(len(labels)) if np.argmax(predictions[i]) == labels[i]]) / len(labels)

def create_model(input_shape, num_classes, dropout_rate):
    model = models.Sequential([
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, train_vectors, train_labels, validation_vectors, validation_labels, epochs, batch_size):
    history = model.fit(train_vectors, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_vectors, validation_labels))
    return history

def save_model(model, file_name):
    model.save(file_name)

def load_model(file_name):
    return tf.keras.models.load_model(file_name)

def predict_labels(model, test_vectors):
    return model.predict(test_vectors)

def plot_training(history_list, model_names):
    num_models = len(history_list)

    for i in range(num_models):
        history = history_list[i]
        model_name = model_names[i]

        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('{} Loss'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('{} Accuracy'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data = load_data_from_csv(DATA_FILE_NAME)

    train_data = data[:40000]
    test_data = data[40000:]

    train_vectors, train_labels = preprocess_data(train_data)
    test_vectors, test_labels = preprocess_data(test_data)

    validation_split = 0.2
    split_index = int(len(train_vectors) * (1 - validation_split))
    validation_vectors, validation_labels = train_vectors[split_index:], train_labels[split_index:]
    train_vectors, train_labels = train_vectors[:split_index], train_labels[:split_index]

    trained_models = []
    predictions_list = []
    history_list = []

    for i, model_file_name in enumerate(NAMES[:-1]):
        model_epochs = MODEL_EPOCHS[i]
        model_batch_size = BATCH_SIZE * (EPOCHS // model_epochs)

        model = create_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.2*(i+1))
        history = train_model(model, train_vectors, train_labels, validation_vectors, validation_labels, model_epochs, model_batch_size)
        save_model(model, model_file_name)
        trained_models.append(model)

        model = load_model(model_file_name)
        predictions = predict_labels(model, test_vectors)
        predictions_list.append(predictions)

        accuracy = get_accuracy(test_labels, predictions)
        print("Model {}: Accuracy: {:.2f}%".format(i+1, accuracy * 100))

        history_list.append(history)

    ultimate_model = create_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.5)
    ultimate_history = train_model(ultimate_model, train_vectors, train_labels, validation_vectors, validation_labels, EPOCHS, BATCH_SIZE)
    save_model(ultimate_model, NAMES[-1])
    trained_models.append(ultimate_model)

    ultimate_model = load_model(NAMES[-1])
    ultimate_predictions = predict_labels(ultimate_model, test_vectors)
    predictions_list.append(ultimate_predictions)

    ultimate_accuracy = get_accuracy(test_labels, ultimate_predictions)
    print("Ultimate Model: Accuracy: {:.2f}%".format(ultimate_accuracy * 100))

    history_list.append(ultimate_history)

    model_names = ["Model 1", "Model 2", "Model 3", "Ultimate Model"]
    plot_training(history_list, model_names)
