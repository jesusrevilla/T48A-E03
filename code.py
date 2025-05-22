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
    """Carga el dataset digits de sklearn
    
    Returns:
        Bunch: Objeto con imágenes y etiquetas de dígitos
    """
    digits = load_digits()
    return digits

# Función para dividir y escalar los datos, test_size=0.2
def dividir_y_escalar_datos(digits):
    """Divide los datos en entrenamiento/prueba y los escala
    
    Args:
        digits: Dataset cargado de sklearn
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) datos divididos y escalados
    """
    # Reformatear imágenes de 8x8 a vectores de 64 elementos
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    
    # Dividir en conjuntos de entrenamiento y prueba (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    """Entrena un clasificador MLP con los parámetros especificados
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
    
    Returns:
        MLPClassifier: Modelo entrenado
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),  # Una capa oculta con 100 neuronas
        max_iter=2000,              # Máximo de iteraciones
        random_state=42,            # Semilla para reproducibilidad
        verbose=False               # No mostrar mensajes de entrenamiento
    )
    mlp.fit(X_train, y_train)
    return mlp

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test, limite_aprobacion=0.85):
    """Evalúa el modelo y verifica si cumple con el límite de aprobación
    
    Args:
        modelo: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        limite_aprobacion: Límite mínimo de precisión requerido
    
    Returns:
        tuple: (accuracy, cumple_aprobacion, y_pred)
    """
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cumple_aprobacion = accuracy >= limite_aprobacion
    return accuracy, cumple_aprobacion, y_pred

# Función para visualizar tres dígitos
def visualizar_digitos(X_test, y_pred, y_test):
    """Muestra 3 dígitos con sus predicciones y etiquetas reales
    
    Args:
        X_test: Datos de prueba
        y_pred: Predicciones del modelo
        y_test: Etiquetas reales
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, image, prediction, label in zip(axes, X_test[:3], y_pred[:3], y_test[:3]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"Predicción: {prediction}\nEtiqueta real: {label}")
    plt.show()

# Pruebas unitarias
class TestDigitClassification(unittest.TestCase):
    
    def setUp(self):
        """Preparación de datos para las pruebas"""
        self.digits = cargar_datos()
        self.X_train, self.X_test, self.y_train, self.y_test = dividir_y_escalar_datos(self.digits)
        self.modelo = entrenar_modelo(self.X_train, self.y_train)
        self.accuracy, self.cumple_aprobacion, _ = evaluar_modelo(
            self.modelo, self.X_test, self.y_test
        )
    
    def test_carga_datos(self):
        """Verifica que el dataset se cargue correctamente"""
        self.assertEqual(len(self.digits.images), len(self.digits.target))
        self.assertEqual(self.digits.images[0].shape, (8, 8))
    
    def test_division_datos(self):
        """Verifica la división de datos en entrenamiento y prueba"""
        self.assertEqual(len(self.X_train) + len(self.X_test), len(self.digits.images))
        self.assertAlmostEqual(len(self.X_test)/len(self.digits.images), 0.2, delta=0.01)
    
    def test_escalado_datos(self):
        """Verifica que los datos se hayan escalado correctamente"""
        self.assertAlmostEqual(np.mean(self.X_train), 0, delta=0.1)
        self.assertAlmostEqual(np.std(self.X_train), 1, delta=0.1)
    
    def test_entrenamiento_modelo(self):
        """Verifica que el modelo se haya entrenado correctamente"""
        self.assertTrue(hasattr(self.modelo, 'coefs_'))
        self.assertTrue(hasattr(self.modelo, 'loss_'))
    
    def test_precision_modelo(self):
        """Verifica que la precisión cumpla con el límite requerido"""
        self.assertGreaterEqual(self.accuracy, 0.85)

# Ejecución del flujo principal
if __name__ == "__main__":
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
    
    # Ejecutar pruebas unitarias
    unittest.main(argv=[''], verbosity=2, exit=False)
