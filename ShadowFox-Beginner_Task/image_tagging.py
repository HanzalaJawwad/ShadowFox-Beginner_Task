# ============================================
# IMAGE TAGGING MODEL USING GPU (TensorFlow)
# ============================================

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# 0. GPU detection and setup
# -------------------------------
print("\n🔍 Checking for available GPUs...\n")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"💪 Using {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected! Using CPU instead.")

# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------
print("\n📦 Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"✅ Dataset loaded: {len(train_images)} training and {len(test_images)} test images.")

# --------------------------------
# 2. Visualize some sample images
# --------------------------------
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.suptitle("Sample Images from CIFAR-10 Dataset")
plt.show()

# ------------------------------
# 3. Build CNN model
# ------------------------------
print("\n🏗️ Building CNN model...")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# -----------------------------
# 4. Compile and train model
# -----------------------------
print("\n⚙️ Compiling and training model...")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with tf.device('/GPU:0'):  # Force training on the GPU if available
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

# ------------------------
# 5. Evaluate model
# ------------------------
print("\n📊 Evaluating model performance...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc:.2f}")

# ----------------------------------
# 6. Visualize training performance
# ----------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ------------------------------
# 7. Make and display prediction
# ------------------------------
print("\n🔮 Making predictions...")
predictions = model.predict(test_images)

num_images = 5
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i])
    predicted_label = class_names[np.argmax(predictions[i])]
    actual_label = class_names[test_labels[i][0]]
    color = 'green' if predicted_label == actual_label else 'red'
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {actual_label}", color=color)
plt.suptitle("Sample Predictions")
plt.show()

# ----------------------------
# 8. Save the trained model
# ----------------------------
model.save("image_tagging_gpu_model.h5")
print("\n💾 Model saved as 'image_tagging_gpu_model.h5'")
