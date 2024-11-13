import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer

# Set up directories
train_dir = '/Users/gobindarora/Downloads/paddy-disease-classification/train_images'
test_dir = '/Users/gobindarora/Downloads/paddy-disease-classification/labeled_test_images'

# Define class names
labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
          'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# Load datasets with validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode="categorical"  # Use categorical for multi-class
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode="categorical"  # Use categorical for multi-class
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=test_dir,
    batch_size=32,
    image_size=(224, 224),
    label_mode="categorical",  # Ensure labels are in categorical format
    shuffle=False
)

# Cache and prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load EfficientNetB0 with frozen layers
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
efficientnet_base.trainable = False

# Build the model
model = Sequential([
    efficientnet_base,
    AveragePooling2D(2, 2),
    Flatten(),
    Dense(220, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')  # Adjusted for 10 classes
])

# Compile the model with categorical crossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping]
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predict on test set
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Convert categorical true labels to integer labels
true_classes = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=labels))

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC and ROC Curve for multi-class
lb = LabelBinarizer()
true_labels_binary = lb.fit_transform(true_classes)
predictions_binary = lb.transform(predicted_classes)

# Calculate AUC for each class
plt.figure(figsize=(12, 8))
for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(true_labels_binary[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve by Class')
plt.legend(loc='lower right')
plt.show()

# Average ROC-AUC
average_roc_auc = roc_auc_score(true_labels_binary, predictions, average="macro")
print(f"\nAverage ROC-AUC Score: {average_roc_auc:.4f}")