# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
# from tensorflow.keras.models import Model

# # === Load Pre-split Training Data ===
# train_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined/train_data.csv")
# X_train = train_df.drop(columns=["Stage"]).values
# y_train = train_df["Stage"].values

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # === Positional Encoding ===
# def positional_encoding(length, d_model):
#     pos = np.arange(length)[:, np.newaxis]
#     i = np.arange(d_model)[np.newaxis, :]
#     angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#     angle_rads = pos * angle_rates
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#     return tf.constant(angle_rads, dtype=tf.float32)

# # === Transformer Block ===
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

# # === Model Definition ===
# def create_transformer_model(input_dim, num_classes):
#     inputs = Input(shape=(input_dim,))
#     x = tf.expand_dims(inputs, -1)  # (batch, features, 1)
#     embed_dim = 64
#     x = Dense(embed_dim)(x)
#     x += positional_encoding(input_dim, embed_dim)
#     transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)
#     x = transformer_block(x)
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation="relu")(x)
#     x = Dropout(0.3)(x)
#     outputs = Dense(num_classes, activation="softmax")(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # === Create Model ===
# # num_encoder_blocks = 1  # 🔧 Try different values like 1, 2, 3, ...
# model = create_transformer_model(input_dim=X_train.shape[1], num_classes=len(set(y_train)))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # === Train ===
# history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# # === Save Model ===
# model.save("/home/yogeshwar/Yogesh-MTP/code/trail_PCA_full/transformer_model_balanced.h5")
# print("\n Model saved successfully.")

# # === Plot Accuracy & Loss ===
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.title('Accuracy Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Loss Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.savefig("/home/yogeshwar/Yogesh-MTP/code/trail_PCA_full/transformer_training_curves_balanced.png")
# plt.show()




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
# from tensorflow.keras.models import Model

# # === Load Pre-split Training Data ===
# train_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/csv/custom_dataset_trans/trans_combined_standard_scaled.csv")
# X_train = train_df.drop(columns=["Stage"]).values
# y_train = train_df["Stage"].values

# # === Transformer Block ===
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

# # === Model Definition ===
# def create_transformer_model(input_dim, num_classes, num_blocks=1):
#     inputs = Input(shape=(input_dim,))
#     x = tf.expand_dims(inputs, -1)  # (batch, features, 1)
#     embed_dim = 64
#     x = Dense(embed_dim)(x)  # Scalar to embedding

#     # Apply multiple Transformer blocks
#     for _ in range(num_blocks):
#         x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)(x)

#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation="relu")(x)
#     x = Dropout(0.3)(x)
#     outputs = Dense(num_classes, activation="softmax")(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # === Create Model ===
# num_encoder_blocks = 6  # 🔧 Try different values like 1, 2, 3, ...
# model = create_transformer_model(input_dim=X_train.shape[1], num_classes=len(set(y_train)), num_blocks=num_encoder_blocks)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # === Train ===
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# # === Save Model ===
# model.save("/home/yogeshwar/Yogesh-MTP/csv/custom_dataset_trans/trans_model_new.h5")
# print("\n Model saved successfully.")

# # === Plot Accuracy & Loss ===
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.title('Accuracy Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Loss Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.savefig("/home/yogeshwar/Yogesh-MTP/csv/custom_dataset_trans/transformer_training_curves.png")
# plt.show()



#For unraveled dataset

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle

# === Multi-GPU Strategy ===
strategy = tf.distribute.MirroredStrategy()
print("GPUs being used:", strategy.num_replicas_in_sync)

# === Load Training Data ===
train_df = pd.read_csv("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/Unraveled_train_data_balanced_standard_scaled.csv")
X_train = train_df.drop(columns=["Stage"]).values
y_train = train_df["Stage"].values

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)

# Save label encoder for testing later
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# === Positional Encoding Function ===
def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads, dtype=tf.float32)

# === Custom Transformer Block ===
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
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
        attn_output = self.att(inputs, inputs)   # Self-attention
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# === Model Builder ===
def create_transformer_model(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    
    # Expand dims to [batch, seq_len, features=1]
    x = Reshape((input_dim, 1))(inputs)
    
    embed_dim = 64
    x = Dense(embed_dim)(x)   # Linear projection to embedding space
    
    # Add positional encoding
    x = x + positional_encoding(input_dim, embed_dim)
    
    # Transformer Block
    x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)(x)
    
    # Global pooling over sequence
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=inputs, outputs=outputs)

# === Train Model inside Multi-GPU scope ===
with strategy.scope():
    model = create_transformer_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_encoded)))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_encoded,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# === Save Model ===
model.save("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/transformer_model.keras")  # use modern .keras format
print("\n✅ Model saved successfully as transformer_model.keras")

# === Save Training Curves ===
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/training_curves.png")
# plt.show()

