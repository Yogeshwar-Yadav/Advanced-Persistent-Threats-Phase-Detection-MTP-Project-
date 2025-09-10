# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
# import tensorflow as tf

# # === Define Positional Encoding (needed for custom layer to work) ===
# def positional_encoding(length, d_model):
#     pos = np.arange(length)[:, np.newaxis]
#     i = np.arange(d_model)[np.newaxis, :]
#     angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#     angle_rads = pos * angle_rates
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#     return tf.constant(angle_rads, dtype=tf.float32)

# # === Recreate the Custom TransformerBlock ===
# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super().__init__()
#         self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = tf.keras.Sequential([
#             Dense(ff_dim, activation='relu'),
#             Dense(embed_dim),
#         ])
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)

#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
#         ffn_output = self.ffn(out1)
#         return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# # === Load Test Data ===
# test_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/custom_dataset_trans/trans_combined_standard_scaled.csv")
# X_test = test_df.drop(columns=["Stage"]).values
# y_test = test_df["Stage"].values

# # === Load Trained Model with Custom Layer ===
# model_path = "/home/yogeshwar/Yogesh-MTP/code/trail_PCA_full/transformer_model_balanced.h5"
# model = load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

# # === Evaluate ===
# loss, acc = model.evaluate(X_test, y_test, verbose=1)
# print(f"\nTest Accuracy: {acc * 100:.2f}%")

# # === Predictions ===
# y_pred = model.predict(X_test).argmax(axis=1)

# # === Classification Report ===
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # === Save Predictions ===
# result_df = pd.DataFrame(X_test)
# result_df["True_Stage"] = y_test
# result_df["Predicted_Stage"] = y_pred
# result_df.to_csv("/home/yogeshwar/Yogesh-MTP/code/trail_PCA_full/transformer_predictions_balanced.csv", index=False)
# print("Predictions saved to transformer_predictions.csv")




#For Unavrled_Dataset
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
# import tensorflow as tf

# # === Transformer Block (must match training) ===
# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = tf.keras.Sequential([
#             Dense(ff_dim, activation='relu'),
#             Dense(embed_dim),
#         ])
#         self.layernorm1 = LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = LayerNormalization(epsilon=1e-6)
#         self.dropout1 = Dropout(rate)
#         self.dropout2 = Dropout(rate)

#     def call(self, inputs, training=None, **kwargs):
#         attn_output = self.att(inputs, inputs, **kwargs)
#         out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
#         ffn_output = self.ffn(out1)
#         return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.att.key_dim,
#             "num_heads": self.att.num_heads,
#             "ff_dim": self.ffn.layers[0].units,
#             "rate": self.dropout1.rate,
#         })
#         return config

# # === Load Test Data ===
# test_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Transformer_New/test_data.csv")
# X_test = test_df.drop(columns=["Stage"]).values
# y_true_labels = test_df["Stage"].values

# # === Load Saved Label Encoder ===
# with open("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Transformer_New/label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# y_test = label_encoder.transform(y_true_labels)
# class_names = label_encoder.classes_

# # === Load Model ===
# model_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Transformer_New/transformer_model_balanced.h5"
# model = load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

# # === Evaluate ===
# loss, acc = model.evaluate(X_test, y_test, verbose=1)
# print(f"\nTest Accuracy: {acc * 100:.2f}%")

# # === Predictions ===
# y_pred = model.predict(X_test).argmax(axis=1)

# # === Classification Report ===
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=class_names))

# # === Confusion Matrix ===
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(cmap='Blues', xticks_rotation=45)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Transformer_New/confusion_matrix_balanced.png")
# plt.show()

# # === Save Predictions to CSV ===
# pred_df = pd.DataFrame(X_test)
# pred_df["True_Stage_Label"] = y_test
# pred_df["True_Stage_Name"] = label_encoder.inverse_transform(y_test)
# pred_df["Predicted_Stage_Label"] = y_pred
# pred_df["Predicted_Stage_Name"] = label_encoder.inverse_transform(y_pred)

# output_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Transformer_New/transformer_predictions_balanced.csv"
# pred_df.to_csv(output_path, index=False)
# print(f"Predictions saved to {output_path}")













# trans_test.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

# === Custom Transformer Block (must be redefined exactly as in training) ===
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# === Load Test Data ===
test_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/Unraveled_test_data_standard_scaled.csv")
X_test = test_df.drop(columns=["Stage"]).values
y_true = test_df["Stage"].values

# === Load saved Label Encoder ===
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Encode true labels
y_test = label_encoder.transform(y_true)
class_names = label_encoder.classes_

# === Load Model (.keras format) ===
model_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/transformer_model.keras"
model = load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

# === Evaluate Model ===
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")

# === Predictions ===
y_pred = model.predict(X_test).argmax(axis=1)

# === Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/confusion_matrix.png")
plt.show()

# === Save Predictions to CSV ===
results = pd.DataFrame(X_test)
results["True_Label"] = y_test
results["True_Stage"] = label_encoder.inverse_transform(y_test)
results["Pred_Label"] = y_pred
results["Pred_Stage"] = label_encoder.inverse_transform(y_pred)

results.to_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/transformer_predictions.csv", index=False)
print("✅ Predictions saved to transformer_predictions.csv")


