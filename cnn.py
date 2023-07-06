import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.regularizers import l2
from matplotlib import pyplot as plt

EPOCHS = 100
BATCH_SIZE = 16
MODEL_EPOCHS = [5, 10, 5, 10, 5, 10]
NAMES = ["model1", "model2", "best_model1", "best_model2", "worst_model1", "worst_model2"]
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
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_worse_model(input_shape, num_classes, dropout_rate):
    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_model2(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
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
        plt.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Training Accuracy')
        plt.plot(np.arange(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('{} Accuracy'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    data = load_data_from_csv(DATA_FILE_NAME)

    train_split = 0.7
    val_split = 0.2
    test_split = 0.1

    split_index_train = int(len(data) * train_split)
    split_index_val = int(len(data) * (train_split + val_split))

    train_data = data[:split_index_train]
    val_data = data[split_index_train:split_index_val]
    test_data = data[split_index_val:]

    train_vectors, train_labels = preprocess_data(train_data)
    val_vectors, val_labels = preprocess_data(val_data)
    test_vectors, test_labels = preprocess_data(test_data)


    trained_models = []
    predictions_list = []
    history_list = []

    for i, model_file_name in enumerate(NAMES[:-2]):
        model_epochs = MODEL_EPOCHS[i]
        model_batch_size = BATCH_SIZE * (EPOCHS // model_epochs)

        model = create_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.2*(i+1))
        history = train_model(model, train_vectors, train_labels, val_vectors, val_labels, model_epochs, model_batch_size)
        save_model(model, model_file_name)
        trained_models.append(model)

        model = load_model(model_file_name)
        predictions = predict_labels(model, test_vectors)
        predictions_list.append(predictions)

        accuracy = get_accuracy(test_labels, predictions)
        print("Model {}: Accuracy: {:.2f}%".format(i+1, accuracy * 100))

        history_list.append(history)

    for i, model_file_name in enumerate(NAMES[-2:]):
        model_epochs = MODEL_EPOCHS[i]
        model_batch_size = BATCH_SIZE * (EPOCHS // model_epochs)

        model = build_model2(input_shape=(28, 28, 1), num_classes=10)
        history = train_model(model, train_vectors, train_labels, val_vectors, val_labels, model_epochs, model_batch_size)
        save_model(model, model_file_name)
        trained_models.append(model)

        model = load_model(model_file_name)
        predictions = predict_labels(model, test_vectors)
        predictions_list.append(predictions)

        accuracy = get_accuracy(test_labels, predictions)
        print("Best Model {}: Accuracy: {:.2f}%".format(i+1, accuracy * 100))

        history_list.append(history)
    
    for i, model_file_name in enumerate(NAMES[-2:]):
        model_epochs = MODEL_EPOCHS[i]
        model_batch_size = BATCH_SIZE * (EPOCHS // model_epochs)

        model = create_worse_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.2*(i+1))
        history = train_model(model, train_vectors, train_labels, val_vectors, val_labels, model_epochs, model_batch_size)
        save_model(model, model_file_name)
        trained_models.append(model)

        model = load_model(model_file_name)
        predictions = predict_labels(model, test_vectors)
        predictions_list.append(predictions)

        accuracy = get_accuracy(test_labels, predictions)
        print("Best Model {}: Accuracy: {:.2f}%".format(i+1, accuracy * 100))

        history_list.append(history)

    model_names = ["Model 1", "Model 2", "Best Model 1", "Best Model 2", "Worst Model 1" , "Worst Model 2"]
    model_names = NAMES[:len(history_list)]
    plot_training(history_list, model_names)