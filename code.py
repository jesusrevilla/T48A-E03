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
    digits = load_digits()
    return digits

# Función para dividir y escalar los datos, test_size=0.2
def dividir_y_escalar_datos(digits):
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    modelo = MLPClassifier(
        hidden_layer_sizes=(100,),  # Una capa oculta con 100 neuronas
        max_iter=2000,             # Máximo de iteraciones
        random_state=42             # Semilla para reproducibilidad
    )
    modelo.fit(X_train, y_train)
    return modelo

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test, limite_aprobacion=0.85):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cumple_aprobacion = accuracy >= limite_aprobacion
    return accuracy, cumple_aprobacion, y_pred

# Función para visualizar tres dígitos
def visualizar_digitos(X_test, y_pred, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, image, prediction, label in zip(axes, X_test[:3], y_pred[:3], y_test[:3]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"Predicción: {prediction}\nEtiqueta real: {label}")
    plt.show()

# Pruebas unitarias
class TestDigitRecognition(unittest.TestCase):
    def setUp(self):
        self.digits = cargar_datos()
        self.X_train, self.X_test, self.y_train, self.y_test = dividir_y_escalar_datos(self.digits)
        self.modelo = entrenar_modelo(self.X_train, self.y_train)
        self.accuracy, self.cumple_aprobacion, _ = evaluar_modelo(self.modelo, self.X_test, self.y_test)
    
    def test_carga_datos(self):
        """Verifica que los datos se carguen correctamente"""
        self.assertEqual(self.digits.data.shape[0], 1797)
        self.assertEqual(self.digits.target.shape[0], 1797)
        self.assertEqual(self.digits.data.shape[1], 64)  # 8x8 píxeles aplanados
    
    def test_division_datos(self):
        """Verifica la división correcta de los datos"""
        # Verificar proporción 80-20
        total_samples = len(self.X_train) + len(self.X_test)
        self.assertAlmostEqual(len(self.X_test)/total_samples, 0.2, delta=0.01)
        
        # Verificar que los datos están escalados
        self.assertAlmostEqual(np.mean(self.X_train), 0, delta=0.01)
        self.assertAlmostEqual(np.std(self.X_train), 1, delta=0.01)
    
    def test_entrenamiento_modelo(self):
        """Verifica que el modelo se entrene correctamente"""
        self.assertTrue(hasattr(self.modelo, 'coefs_'))  # Verifica que tenga los pesos aprendidos
        self.assertTrue(hasattr(self.modelo, 'n_iter_'))  # Verifica que haya completado iteraciones
    
    def test_precision_modelo(self):
        """Verifica que la precisión del modelo sea aceptable"""
        self.assertGreaterEqual(self.accuracy, 0.85)
        self.assertTrue(self.cumple_aprobacion)

# Ejecución del flujo principal
if __name__ == "__main__":
    # Ejecutar pruebas unitarias
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n" + "="*50 + "\n")
    
    # Ejecutar el flujo de reconocimiento de dígitos
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
