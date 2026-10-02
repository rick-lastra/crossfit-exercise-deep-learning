"""Python code exported from the course project workflow notebook. Review paths and data access before use."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

# Configuración de hiperparámetros (Experiment Tracking)
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Estandarización obligatoria para modelos preentrenados
EPOCHS = 10
LEARNING_RATE = 1e-4 # Tasa baja para ajuste fino (fine-tuning)

# 1. Carga de datos con transformación a tensores
path = kagglehub.dataset_download("hasyimabdillah/workoutexercises-images")

train_ds = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# 2. Pipeline de Preprocesamiento Integrado (Estandarización y Aumento)
preprocessing_model = keras.Sequential([
    layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# 3. Combate al Desbalance (Pesos de Clase)
class_names = train_ds.class_names
labels = np.concatenate([y for x, y in train_ds], axis=0).argmax(axis=1)
counts = np.bincount(labels)
class_weights = {i: len(labels) / (len(class_names) * count) for i, count in enumerate(counts)}

def create_model(base_arch='resnet'):
    # Selección de arquitectura preentrenada
    if base_arch == 'resnet':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    base.trainable = False # Congelar base para Transfer Learning

    model = models.Sequential([
        preprocessing_model,
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 1. Entrenamiento de ResNet50
print("Iniciando entrenamiento de ResNet50...")
model_resnet = create_model('resnet')
history_resnet = model_resnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# 2. Entrenamiento de VGG16
print("\nIniciando entrenamiento de VGG16...")
model_vgg = create_model('vgg')
history_vgg = model_vgg.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

def plot_learning_curves(history, name="Modelo"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Gráfica de Precisión (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.title(f'Precisión: {name}')
    plt.legend()

    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.title(f'Pérdida: {name}')
    plt.legend()
    plt.show()

# Visualización de la Matriz de Confusión genérica
def graficar_matriz_confusion(model, ds, class_names, name="Modelo"):
    # Obtención de predicciones
    y_true = []
    y_pred = []
    for x, y in ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(model.predict(x), axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión: {name} (project workflow)')
    plt.ylabel('Clase Real')
    plt.xlabel('Predicción')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Reporte de Clasificación
    print(f"\nReporte de Clasificación Detallado - project workflow ({name}):")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Ejecución de comparativa visual
plot_learning_curves(history_resnet, name="ResNet50")
plot_learning_curves(history_vgg, name="VGG16")

graficar_matriz_confusion(model_resnet, val_ds, class_names, name="ResNet50")
graficar_matriz_confusion(model_vgg, val_ds, class_names, name="VGG16")

