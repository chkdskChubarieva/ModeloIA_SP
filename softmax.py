import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# --- 1. Leer datos de entrenamiento desde CSV ---
df_train = pd.read_csv('usuarios_marketing.csv')

# --- 2. Separar características y etiqueta ---
X_train_raw = df_train.drop('clase_usuario', axis=1)
y_train_raw = df_train['clase_usuario']

# Codificar etiquetas a números
clases = y_train_raw.unique()
clase_to_num = {c: i for i, c in enumerate(clases)}
y_train_num = y_train_raw.map(clase_to_num)

# One-hot encoding para salida
y_train_cat = to_categorical(y_train_num)

# --- 3. Definir columnas categóricas y numéricas ---
cat_features = ['ubicacion', 'categoria_productos']
num_features = ['frecuencia_visitas', 'recencia', 'interacciones', 'tipo_dispositivo']

# --- 4. Crear pipeline de preprocesamiento ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# Ajustar preprocesador con datos de entrenamiento
X_train = preprocessor.fit_transform(X_train_raw)

# --- 5. Definir modelo ---
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(12, activation='relu'),
    Dense(len(clases), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Entrenar modelo ---
model.fit(X_train, y_train_cat, epochs=50, batch_size=4, verbose=1)

# --- 7. Leer CSV de usuarios nuevos ---
df_nuevos = pd.read_csv('usuarios_nuevos.csv')

# --- 8. Preprocesar datos nuevos usando el pipeline entrenado ---
X_nuevos = preprocessor.transform(df_nuevos)

# --- 9. Predecir probabilidades y clases ---
pred_prob = model.predict(X_nuevos)
pred_clases_num = np.argmax(pred_prob, axis=1)

# Mapear a etiquetas originales
num_to_clase = {v: k for k, v in clase_to_num.items()}
pred_clases = [num_to_clase[num] for num in pred_clases_num]

# --- 10. Añadir la columna con la predicción ---
df_nuevos['clase_predicha'] = pred_clases

# --- 11. Mostrar resultado final ---
print(df_nuevos)
