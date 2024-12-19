import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

assert X_train.shape == (50000, 32, 32, 3)
assert X_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# convert to 1D array
print("Before:\n", y_train[:5])

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print("After:\n", y_train[:5])

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


plot_sample(X_train, y_train, 7)

# Normalize the data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# --- Artificial Neural Network (ANN) ---
ann = models.Sequential(
    [
        layers.Flatten(
            input_shape=(32, 32, 3)
        ),  # Flatten the 32x32x3 input image into a 1D array
        layers.Dense(3000, activation="relu"),  # 3000 neurons, ReLU activation
        layers.Dense(1000, activation="relu"),
        layers.Dense(
            10, activation="softmax"
        ),  # Output layer with 10 neurons (one for each class) and softmax activation
    ]
)

ann.compile(
    optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

ann.fit(X_train, y_train, epochs=5)

y_pred = ann.predict(X_test)

y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, y_pred_classes), annot=True, fmt="d")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()

# --- Convolutional Neural Network (CNN) ---
cnn = models.Sequential(
    [
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

cnn.fit(X_train, y_train, epochs=8)

cnn.evaluate(X_test, y_test)

y_pred = cnn.predict(X_test)
y_pred[:5]

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]

# Classification Report
print("Classification Report: \n", classification_report(y_test, y_classes))

plot_sample(X_test, y_classes, 7)
plot_sample(X_test, y_test, 7)

plot_sample(X_test, y_classes, 10)
plot_sample(X_test, y_test, 10)

# --- Transfer Learning with VGG16 ---

# Create VGG16 base model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False

# Create new model on top
transfer_model = models.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile model
transfer_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
)

# Train with data augmentation
history = transfer_model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test),
)

# Evaluate and plot results
y_pred = transfer_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(
    "\nTransfer Learning Model Classification Report:\n",
    classification_report(y_test, y_pred_classes),
)

# Test
plot_sample(X_test, y_test, 7)
plot_sample(X_test, y_pred_classes, 7)

plot_sample(X_test, y_test, 10)
plot_sample(X_test, y_pred_classes, 10)
