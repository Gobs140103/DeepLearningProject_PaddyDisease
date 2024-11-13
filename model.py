import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer

train_path = '/Users/gobindarora/Downloads/paddy-disease-classification/train_images'

test_path = '/Users/gobindarora/Downloads/paddy-disease-classification/labeled_test_images'

classes = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
           'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path, batch_size=32, 
    image_size=(224, 224), 
    validation_split=0.2, 
    subset="training",
    seed=123, 
    label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_path, 
    batch_size=32, 
    image_size=(224, 224), 
    validation_split=0.2, 
    subset="validation",
    seed=123, 
    label_mode="categorical"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path, 
    batch_size=32, 
    image_size=(224, 224), 
    label_mode="categorical", 
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    AveragePooling2D(2, 2),
    Flatten(),
    Dense(220, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    'best_paddy_disease_classifier.keras', monitor='val_accuracy', save_best_only=True, verbose=1
)

history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=40,
    callbacks=[early_stop, checkpoint])

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

preds = model.predict(test_ds)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)

print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=classes))

conf_mat = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

lb = LabelBinarizer()
true_bin = lb.fit_transform(true_classes)
pred_bin = lb.transform(pred_classes)

plt.figure(figsize=(12, 8))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(true_bin[:, i], preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve by Class')
plt.legend(loc='lower right')
plt.show()

avg_auc = roc_auc_score(true_bin, preds, average="macro")
print(f"\nAverage ROC-AUC Score: {avg_auc:.4f}")

model.save('final_paddy_disease_classifier.keras')