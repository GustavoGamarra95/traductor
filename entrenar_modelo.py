import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Leer el archivo CSV con la codificación correcta
try:
    df = pd.read_csv('movimientos_manos.csv', encoding='latin-1')  # Cambia 'latin-1' por la codificación correcta
except UnicodeDecodeError:
    # Si falla, intenta con 'utf-8' y maneja errores
    df = pd.read_csv('movimientos_manos.csv', encoding='utf-8', errors='replace')

# Verificar las primeras filas del DataFrame
print("Datos originales:")
print(df.head())

# Separar características (X) y etiquetas (y)
y = df['Accion']  # Etiquetas (columna 'Accion')
X = df.drop(columns=['Accion'])  # Características (todas las columnas excepto 'Accion')

# Identificar columnas numéricas y categóricas
columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns
columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns

print("\nColumnas numéricas:", columnas_numericas.tolist())
print("Columnas categóricas:", columnas_categoricas.tolist())

# Codificar columnas categóricas (si existen)
if len(columnas_categoricas) > 0:
    print("\nCodificando columnas categóricas...")
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Usar OneHotEncoder para columnas categóricas
    encoded_categorical = encoder.fit_transform(X[columnas_categoricas])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(columnas_categoricas))

    # Combinar columnas numéricas y codificadas
    X = pd.concat([X[columnas_numericas], encoded_categorical_df], axis=1)
else:
    print("\nNo hay columnas categóricas para codificar.")

# Escalar las características numéricas
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Codificar las etiquetas (si son categóricas)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
print("\nEntrenando el modelo...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo y los objetos de preprocesamiento
model.save('modelo_movimientos_manos.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
if len(columnas_categoricas) > 0:
    joblib.dump(encoder, 'onehot_encoder.pkl')  # Guardar el OneHotEncoder si se usó

print("\nModelo entrenado y guardado correctamente.")