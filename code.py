import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Función para cargar el dataset
def cargar_datos():
    pass

# Función para dividir y escalar los datos, test_size=0.2
def dividir_y_escalar_datos(digits):
    pass

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    pass

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test, limite_aprobacion=0.85):
    pass

# Función para visualizar tres dígitos
def visualizar_digitos(X_test, y_pred, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, image, prediction, label in zip(axes, X_test[:3], y_pred[:3], y_test[:3]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"Predicción: {prediction}\nEtiqueta real: {label}")
    plt.show()


# Ejecución del flujo principal
digits = cargar_datos()
X_train, X_test, y_train, y_test = dividir_y_escalar_datos(digits)
modelo = entrenar_modelo(X_train, y_train)
accuracy, cumple_aprobacion, y_pred = evaluar_modelo(modelo, X_test, y_test)

print(f"Precisión del modelo: {accuracy * 100:.2f}%")
if cumple_aprobacion:
    print("✅ El modelo cumple con el límite de aprobación.")
else:
    print("❌ El modelo no cumple con el límite de aprobación.")

visualizar_digitos(X_test, y_pred, y_test)
