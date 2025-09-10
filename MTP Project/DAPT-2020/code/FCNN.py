import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------------- Load Data -----------------------
df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/pca_csv/pca_50_components.csv")

# Separate features and labels
X = df.drop(columns=["Stage"]).values
y = df["Stage"].values

# Encode labels if they are not numeric
if df["Stage"].dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# -------------------- Stratified Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Compute Class Weights -------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("\nComputed Class Weights:", class_weight_dict)

# -------------------- Build FCNN Model -------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')  # Multi-class
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------- Train Model -------------------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight_dict  # Apply class weights here
)

# -------------------- Evaluate Model -------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\Test Accuracy: {acc * 100:.2f}%")

# -------------------- Classification Report -------------------
y_pred = model.predict(X_test).argmax(axis=1)
print("\Classification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/home/yogeshwar/Yogesh-MTP/Models/FCNN/fcnn_accuracy_loss_plot_50.png')
plt.show()


# -------------------- Save Model -------------------
model.save("/home/yogeshwar/Yogesh-MTP/Models/FCNN/fcnn_model_pca_50.h5")
print("\n Model saved to 'fcnn_model_pca_50.h5'")
