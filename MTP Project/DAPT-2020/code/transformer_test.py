import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
import tensorflow as tf

# === Define Positional Encoding (needed for custom layer to work) ===
def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads, dtype=tf.float32)

# === Recreate the Custom TransformerBlock ===
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# === Load Test Data ===
test_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Models/Transformer/pca_50_components_test_data.csv")
X_test = test_df.drop(columns=["Stage"]).values
y_test = test_df["Stage"].values

# === Load Trained Model with Custom Layer ===
model_path = "/home/yogeshwar/Yogesh-MTP/Models/Transformer/transformer_model_pca50.h5"
model = load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

# === Evaluate ===
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# === Predictions ===
y_pred = model.predict(X_test).argmax(axis=1)

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Save Predictions ===
result_df = pd.DataFrame(X_test)
result_df["True_Stage"] = y_test
result_df["Predicted_Stage"] = y_pred
result_df.to_csv("/home/yogeshwar/Yogesh-MTP/Models/Transformer/transformer_predictions.csv", index=False)
print("Predictions saved to transformer_predictions.csv")