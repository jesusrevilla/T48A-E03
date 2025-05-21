import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from code import (cargar_datos, dividir_y_escalar_datos, entrenar_modelo, evaluar_modelo)

# Pruebas unitarias
class TestClasificacionDigits(unittest.TestCase):
    def setUp(self):
        self.digits = cargar_datos()
        self.X_train, self.X_test, self.y_train, self.y_test = dividir_y_escalar_datos(self.digits)
        self.modelo = entrenar_modelo(self.X_train, self.y_train)

    def test_cargar_datos(self):
        self.assertEqual(len(self.digits.data), 1797)

    def test_dividir_y_escalar_datos(self):
        self.assertEqual(self.X_train.shape[0], 1437)
        self.assertEqual(self.X_test.shape[0], 360)

    def test_entrenar_modelo(self):
        self.assertIsInstance(self.modelo, MLPClassifier)

    def test_evaluar_modelo(self):
        accuracy, cumple_aprobacion, y_pred = evaluar_modelo(self.modelo, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.85)
        self.assertEqual(len(y_pred), len(self.y_test))
      
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
