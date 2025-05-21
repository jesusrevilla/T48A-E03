# T48A-E03
T48A-E03 Examen del tercer parcial

# Clasificación de Dígitos con Red de Perceptrones (MLPClassifier)

Este proyecto utiliza el dataset `digits` de `sklearn` para entrenar una red neuronal multicapa (MLP) que clasifica imágenes de dígitos escritos a mano.

---

## 🧠 Flujo de Trabajo

### 1. Carga del Dataset
- Se utiliza `load_digits()` para obtener imágenes de dígitos (0–9).
- Cada imagen es de 8x8 píxeles con su etiqueta correspondiente.

### 2. Preprocesamiento
- División del dataset en entrenamiento (80%) y prueba (20%) con `train_test_split`.
- Escalado de características con `StandardScaler` para mejorar la convergencia del modelo.

### 3. Entrenamiento del Modelo
- Se entrena un `MLPClassifier` con:
  - 100 neuronas en la capa oculta.
  - Hasta 2000 iteraciones (`max_iter=2000`).
  - Semilla aleatoria (`random_state=42`) para reproducibilidad.

### 4. Evaluación del Modelo
- Se calcula la **precisión** del modelo en el conjunto de prueba.
- Se verifica si la precisión supera un **límite de aprobación del 85%**.
- Se obtienen las predicciones del modelo para análisis posterior.

### 5. Visualización
- Se muestran 3 imágenes del conjunto de prueba con:
  - La predicción del modelo.
  - La etiqueta real del dígito.

### 6. Pruebas Unitarias
- Se implementan pruebas con `unittest` para validar:
  - Carga correcta del dataset.
  - División y escalado adecuados.
  - Entrenamiento exitoso del modelo.
  - Precisión mínima aceptable (≥ 85%).

---

## ✅ Requisitos
- Python 3.7+
- scikit-learn
- matplotlib
- numpy

---

## 🚀 Ejecución
Puedes ejecutar este proyecto en Google Colab o en tu entorno local. Asegúrate de tener instaladas las dependencias necesarias.


