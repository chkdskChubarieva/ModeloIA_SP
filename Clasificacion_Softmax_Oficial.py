#!/usr/bin/env python
# coding: utf-8

# 1. LIBRERÍAS
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import tkinter as tk
from tkinter import filedialog

# 2. SELECCIÓN DE ARCHIVO CSV
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV Files", ".csv"), ("Todos los archivos", ".*")])
df = pd.read_csv(file_path, encoding='utf-8')

# 3. PREPARACIÓN DE DATOS
X_raw = df.drop('clase_usuario', axis=1)
y_raw = df['clase_usuario']

clases = np.sort(y_raw.unique())
clase_to_num = {c:i for i,c in enumerate(clases)}
num_to_clase = {i:c for c,i in clase_to_num.items()}
y_num = y_raw.map(clase_to_num)
y_cat = np.eye(len(clases))[y_num]

# Columnas numéricas y categóricas
num_feats = ['frecuencia_visitas', 'recencia', 'interacciones']
cat_feats = ['ubicacion', 'categoria_productos','tipo_dispositivo']

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats)
])

# Separar entrenamiento y validación antes del escalado
X_train_raw, X_val_raw, y_train_cat, y_val_cat, y_train_num, y_val_num = train_test_split(
    X_raw, y_cat, y_num, test_size=0.2, stratify=y_num, random_state=42
)

# Ajustar preprocesador con datos de entrenamiento solamente
X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)

# Conversión a tensores
X_train = tf.constant(X_train, dtype=tf.float32)
y_train = tf.constant(y_train_cat, dtype=tf.float32)
X_val = tf.constant(X_val, dtype=tf.float32)
y_val = tf.constant(y_val_cat, dtype=tf.float32)

# 4. INICIALIZACIÓN DE PARÁMETROS
num_features = X_train.shape[1]
num_classes = y_train.shape[1]
tf.random.set_seed(42)  # Semilla reproducible

W = tf.Variable(tf.random.normal(shape=(num_features, num_classes), stddev=0.01), name='weights')
b = tf.Variable(tf.zeros(shape=(num_classes,)), name='bias')

# 5. MODELO LINEAL + SOFTMAX
def model(X):
    logits = tf.matmul(X, W) + b
    return tf.nn.softmax(logits)

# 6. FUNCIÓN DE COSTO (Entropía Cruzada + Regularización L2)
def cross_entropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)
    ce_loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    l2_loss = 0.01 * tf.nn.l2_loss(W)
    return ce_loss + l2_loss

# 7. MÉTRICA DE PRECISIÓN
def accuracy(y_true, y_pred):
    correct = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# 8. ENTRENAMIENTO
optimizer = tf.optimizers.Adam(learning_rate=0.001)
epochs = 200
batch_size = 32
patience = 10
best_val_loss = np.inf
wait = 0

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)

# Guardado de los mejores pesos
best_W, best_b = None, None

for epoch in range(1, epochs + 1):
    epoch_loss, epoch_acc, batches = 0, 0, 0
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            preds = model(batch_x)
            loss = cross_entropy(batch_y, preds)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))

        epoch_loss += loss.numpy()
        epoch_acc += accuracy(batch_y, preds).numpy()
        batches += 1

    epoch_loss /= batches
    epoch_acc /= batches

    val_preds = model(X_val)
    val_loss = cross_entropy(y_val, val_preds).numpy()
    val_acc = accuracy(y_val, val_preds).numpy()

    print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_W = W.numpy()
        best_b = b.numpy()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("📌 Entrenamiento detenido por early stopping.")
            break

# Restaurar los mejores pesos
W.assign(best_W)
b.assign(best_b)

# 9. PREDICCIÓN SOBRE NUEVOS USUARIOS
df_new = pd.read_csv('usuarios_nuevos.csv')
X_new = preprocessor.transform(df_new)
X_new = tf.constant(X_new, dtype=tf.float32)

pred_prob = model(X_new).numpy()
pred_num = np.argmax(pred_prob, axis=1)
df_new['clase_predicha'] = [num_to_clase[n] for n in pred_num]

print("\n📊 Clasificación de nuevos usuarios:")
print(df_new)

# 10. GRÁFICA EN GRADIENTE (Heatmap)
plt.figure(figsize=(12, min(1 + 0.5 * len(df_new), 12)))  # Ajusta alto según cantidad de usuarios

# Crear DataFrame para el heatmap
heatmap_df = pd.DataFrame(pred_prob, columns=clases)
heatmap_df.index = [f"Usuario {i+1}" for i in range(len(df_new))]

# Graficar el mapa de calor
sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt=".2f", cbar_kws={"label": "Probabilidad"})
plt.title("Probabilidades de clasificación por clase (Softmax)")
plt.xlabel("Clase")
plt.ylabel("Nuevos usuarios")
plt.tight_layout()
plt.show()
