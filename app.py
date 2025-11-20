import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import resample
import requests
from PIL import Image
import io
import urllib.request
import json
from collections import Counter
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class CaptchaBreaker:
    def __init__(self, target_size=(32, 32)):
        """
        Inicializa el sistema de ruptura de captchas
        
        Args:
            target_size: Tamaño objetivo para normalizar las imágenes
        """
        self.target_size = target_size
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}
        self.images = []
        self.labels = []
        self.features = []
        self.stats = {
            'total_images': 0,
            'characters_found': 0,
            'segmented_chars': 0,
            'unique_chars': 0
        }
    
    def load_captcha_images_from_folder(self, folder_path):
        """
        Componente 1: Adquisición de datos desde una carpeta local
        
        ENTRADA: Ruta de la carpeta con imágenes de captcha
        SALIDA: Lista de imágenes cargadas
        
        Carga imágenes de captcha desde una carpeta local.
        """
        print(f"Cargando imágenes de captcha desde: {folder_path}...")
        downloaded_images = []
        
        if not os.path.isdir(folder_path):
            print(f"Error: La carpeta '{folder_path}' no existe.")
            return []

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pbm'))]
        
        for i, filename in enumerate(image_files):
            try:
                filepath = os.path.join(folder_path, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    downloaded_images.append(img)
                else:
                    print(f"Advertencia: No se pudo cargar la imagen {filename}. Posiblemente dañada o formato no soportado.")
            except Exception as e:
                print(f"Error al cargar la imagen {filename}: {e}")
            
            if i % 100 == 0 and i > 0:
                print(f"Cargadas {i+1}/{len(image_files)} imágenes")
        
        self.stats['total_images'] = len(downloaded_images)
        print(f"Carga completada: {len(downloaded_images)} imágenes")
        return downloaded_images
    
    
    def binarize_image(self, image, threshold=127):
        """
        Componente 2: Binarización de imágenes
        
        ENTRADA: Imagen RGB, umbral de binarización
        SALIDA: Imagen binaria
        
        Convierte una imagen RGB a binaria usando umbralización
        """
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Aplicar umbralización
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Invertir si es necesario (texto negro sobre fondo blanco)
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        return binary
    
    def segment_characters(self, binary_image):
        """
        Componente 3: Segmentación de caracteres
        
        ENTRADA: Imagen binaria
        SALIDA: Lista de imágenes de caracteres individuales
        
        Segmenta caracteres individuales usando análisis de contornos
        """
        # Encontrar contornos
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área y ratio de aspecto
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h
            
            # Filtros para caracteres válidos
            if area > 50 and 0.2 < aspect_ratio < 2.0 and h > 10 and w > 5:
                valid_contours.append((x, y, w, h))
        
        # Ordenar contornos de izquierda a derecha
        valid_contours.sort(key=lambda x: x[0])
        
        # Extraer caracteres individuales
        characters = []
        for x, y, w, h in valid_contours:
            char_img = binary_image[y:y+h, x:x+w]
            characters.append(char_img)
        
        self.stats['segmented_chars'] = len(characters)
        return characters
    
    def normalize_character_size(self, char_image):
        """
        Componente 4: Normalización de tamaño
        
        ENTRADA: Imagen de carácter
        SALIDA: Imagen normalizada en marco fijo
        
        Coloca el carácter en un marco de tamaño fijo manteniendo proporciones
        """
        # Crear marco vacío
        frame = np.zeros(self.target_size, dtype=np.uint8)
        
        if char_image.size == 0:
            return frame
        
        # Calcular escalado manteniendo proporciones
        h, w = char_image.shape
        scale = min(self.target_size[0] / h, self.target_size[1] / w) * 0.8
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h > 0 and new_w > 0:
            # Redimensionar
            resized = cv2.resize(char_image, (new_w, new_h))
            
            # Centrar en el marco
            start_h = (self.target_size[0] - new_h) // 2
            start_w = (self.target_size[1] - new_w) // 2
            
            frame[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        
        return frame
    
    def extract_features(self, image):
        """
        Componente 5: Extracción de características
        
        ENTRADA: Imagen normalizada
        SALIDA: Vector de características 1D
        
        Extrae características de la imagen: histograma, momentos, y píxeles
        """
        features = []
        
        # Características básicas: píxeles flatten
        pixels = image.flatten() / 255.0
        features.extend(pixels)
        
        # Histograma
        hist = cv2.calcHist([image], [0], None, [16], [0, 256])
        features.extend(hist.flatten())
        
        # Momentos de Hu
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)
        
        # Características de textura (LBP simplificado)
        lbp_features = self._calculate_lbp_features(image)
        features.extend(lbp_features)
        
        return np.array(features)
    
    def _calculate_lbp_features(self, image):
        """
        Calcula características LBP simplificadas
        """
        # Implementación simplificada de LBP
        h, w = image.shape
        lbp_features = []
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                pattern = 0
                
                # Comparar con vecinos
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern += 2**k
                
                lbp_features.append(pattern)
        
        # Histograma de patrones LBP
        hist, _ = np.histogram(lbp_features, bins=32, range=(0, 256))
        return hist / (hist.sum() + 1e-7)
    
    def cluster_characters(self, features, n_clusters=None):
        """
        Componente 6: Clustering para etiquetado automático
        
        ENTRADA: Matriz de características, número de clusters
        SALIDA: Etiquetas de cluster
        
        Agrupa caracteres similares usando K-means
        """
        if n_clusters is None:
            # Estimar número de clusters usando método del codo
            n_clusters = self._estimate_clusters(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        self.stats['unique_chars'] = n_clusters
        return cluster_labels
    
    def _estimate_clusters(self, features):
        """
        Estima el número óptimo de clusters usando el método del codo
        """
        max_clusters = min(20, len(features) // 10)
        if max_clusters < 2:
            return 2
        
        inertias = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Encontrar el codo
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diff2 = np.diff(diffs)
            if len(diff2) > 0:
                elbow = np.argmax(diff2) + 2
                return min(elbow, max_clusters)
        
        return min(10, max_clusters)
    
    def reduce_dimensionality(self, features, method='pca', n_components=0.95):
        """
        Componente 7: Reducción de dimensionalidad
        
        ENTRADA: Matriz de características, método, número de componentes
        SALIDA: Características reducidas
        
        Reduce la dimensionalidad usando PCA
        """
        if method == 'pca':
            self.pca = PCA(n_components=n_components)
            reduced_features = self.pca.fit_transform(features)
            
            print(f"Dimensionalidad original: {features.shape[1]}")
            print(f"Dimensionalidad reducida: {reduced_features.shape[1]}")
            print(f"Varianza explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
            
            return reduced_features
        
        elif method == 'average_pooling':
            # Reducción por promedio (para imágenes)
            return self._average_pooling_reduction(features)
        
        return features
    
    def _average_pooling_reduction(self, features):
        """
        Reducción por promedio de píxeles
        """
        # Asumir que las primeras target_size[0]*target_size[1] características son píxeles
        pixel_count = self.target_size[0] * self.target_size[1]
        
        if features.shape[1] >= pixel_count:
            # Reshape a imagen y aplicar pooling
            pixel_features = features[:, :pixel_count]
            other_features = features[:, pixel_count:]
            
            # Reshape a imagen
            images = pixel_features.reshape(-1, self.target_size[0], self.target_size[1])
            
            # Average pooling 2x2
            pooled_images = []
            for img in images:
                pooled = cv2.resize(img, (self.target_size[1]//2, self.target_size[0]//2))
                pooled_images.append(pooled.flatten())
            
            pooled_features = np.array(pooled_images)
            
            # Concatenar con otras características
            if other_features.shape[1] > 0:
                return np.hstack([pooled_features, other_features])
            else:
                return pooled_features
        
        return features
    
    def balance_dataset(self, features, labels):
        """
        Componente 8: Balanceo de datos
        
        ENTRADA: Características, etiquetas
        SALIDA: Características y etiquetas balanceadas
        
        Balancea el dataset usando SMOTE
        """
        print("Distribución original:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Clase {label}: {count} ejemplos")
        
        # Graficar distribución original
        plt.figure(figsize=(10, 5))
        sns.barplot(x=unique, y=counts)
        plt.title('Distribución de Clases Original')
        plt.xlabel('Clase')
        plt.ylabel('Número de Ejemplos')
        plt.show()

        # Aplicar SMOTE
        smote = SMOTE(random_state=42)
        balanced_features, balanced_labels = smote.fit_resample(features, labels)
        
        print("\nDistribución después del balanceo:")
        unique, counts = np.unique(balanced_labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Clase {label}: {count} ejemplos")

        # Graficar distribución después del balanceo
        plt.figure(figsize=(10, 5))
        sns.barplot(x=unique, y=counts)
        plt.title('Distribución de Clases Después de SMOTE')
        plt.xlabel('Clase')
        plt.ylabel('Número de Ejemplos')
        plt.show()
        
        return balanced_features, balanced_labels
    
    def train_models(self, features, labels):
        """
        Componente 9: Entrenamiento de modelos
        
        ENTRADA: Características, etiquetas
        SALIDA: Modelos entrenados
        
        Entrena múltiples clasificadores
        """
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definir modelos
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=1000, 
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, 
                random_state=42
            )
        }
        
        # Entrenar modelos
        results = {}
        for name, model in models.items():
            print(f"\nEntrenando {name}...")
            
            # Entrenar
            model.fit(X_train_scaled, y_train)
            
            # Evaluar
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                # Removed 'X_test_scaled' and 'y_test' from here to prevent redundancy,
                # as they are already returned as 'test_data' tuple.
                # The evaluate_model and plotting functions will use the 'test_data' tuple directly.
            }
            
            print(f"Precisión entrenamiento: {train_score:.3f}")
            print(f"Precisión prueba: {test_score:.3f}")
            print(f"Validación cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models = results
        return results, (X_test_scaled, y_test) # test_data is (X_test_scaled, y_test)
    
    def evaluate_model(self, model, X_test, y_test, class_names):
        """
        Componente 10: Evaluación del modelo
        
        ENTRADA: Modelo, datos de prueba
        SALIDA: Métricas de evaluación
        
        Evalúa el rendimiento del modelo
        """
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Precisión: {accuracy:.3f}")
        print(f"\nReporte de clasificación:\n{report}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test,
            'class_names': class_names
        }

    def plot_confusion_matrix(self, cm, class_names, title="Matriz de Confusión"):
        """
        Grafica la matriz de confusión.
        
        Args:
            cm: La matriz de confusión (numpy array).
            class_names: Lista de nombres de las clases (strings).
            title: Título para el gráfico.
        """
        plt.figure(figsize=(len(class_names) + 2, len(class_names) + 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel("Etiqueta Predicha")
        plt.ylabel("Etiqueta Verdadera")
        plt.show()

    def plot_roc_curve(self, model, X_test, y_test, class_names, title="Curva ROC"):
        """
        Grafica la curva ROC para modelos de clasificación multiclase (OvA).
        
        Args:
            model: El modelo entrenado.
            X_test: Características de prueba.
            y_test: Etiquetas verdaderas de prueba.
            class_names: Nombres de las clases.
            title: Título para el gráfico.
        """
        if not hasattr(model, 'predict_proba'):
            print(f"El modelo {type(model).__name__} no soporta predict_proba. No se puede generar la curva ROC.")
            return

        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_binarized.shape[1]

        if n_classes == 1:
            print("Solo una clase presente en y_test, no se puede generar la curva ROC.")
            return

        y_score = model.predict_proba(X_test)

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Clase {class_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, model, X_test, y_test, class_names, title="Curva Precisión-Recall"):
        """
        Grafica la curva Precisión-Recall para modelos de clasificación multiclase (OvA).
        
        Args:
            model: El modelo entrenado.
            X_test: Características de prueba.
            y_test: Etiquetas verdaderas de prueba.
            class_names: Nombres de las clases.
            title: Título para el gráfico.
        """
        if not hasattr(model, 'predict_proba'):
            print(f"El modelo {type(model).__name__} no soporta predict_proba. No se puede generar la curva Precisión-Recall.")
            return

        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_binarized.shape[1]

        if n_classes == 1:
            print("Solo una clase presente en y_test, no se puede generar la curva Precisión-Recall.")
            return

        y_score = model.predict_proba(X_test)

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
            avg_precision = average_precision_score(y_test_binarized[:, i], y_score[:, i])
            plt.plot(recall, precision, label=f'Clase {class_names[i]} (AP = {avg_precision:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precisión')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.show()

    def plot_feature_importance(self, model, feature_names=None, title="Importancia de las Características"):
        """
        Grafica la importancia de las características para modelos basados en árboles.
        
        Args:
            model: El modelo entrenado (RandomForestClassifier o DecisionTreeClassifier).
            feature_names: Nombres de las características (lista de strings).
            title: Título para el gráfico.
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"El modelo {type(model).__name__} no tiene el atributo feature_importances_.")
            return

        importances = model.feature_importances_
        if feature_names is None:
            # If feature_names are not provided, create generic ones
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_importance = df_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        # Ensure we don't try to plot more features than available
        top_n = min(20, len(df_importance))
        sns.barplot(x='Importance', y='Feature', data=df_importance.head(top_n)) # Mostrar top 20
        plt.title(title)
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.show()
    
    def process_captcha_pipeline(self, folder_path):
        """
        Pipeline completo para procesar captchas
        
        ENTRADA: Ruta de la carpeta con imágenes de captcha
        SALIDA: Modelo entrenado y métricas
        """
        print("=== INICIANDO PIPELINE DE PROCESAMIENTO DE CAPTCHAS ===\n")
        
        # 1. Adquisición de datos
        print("1. Adquisición de datos...")
        captcha_images = self.load_captcha_images_from_folder(folder_path)
        
        if not captcha_images:
            print("No se encontraron imágenes en la carpeta especificada. Saliendo del pipeline.")
            return None, None

        # Separar 10% para validación final
        validation_split = int(len(captcha_images) * 0.1)
        validation_images = captcha_images[:validation_split]
        training_images = captcha_images[validation_split:]
        
        # 2. Procesamiento de imágenes
        print("\n2. Procesamiento de imágenes...")
        all_features = []
        all_labels = [] # Las etiquetas serán asignadas por el clustering
        
        for i, img in enumerate(training_images):
            # Binarizar
            binary = self.binarize_image(img)
            
            # Segmentar caracteres
            characters = self.segment_characters(binary)
            
            # Procesar cada carácter
            for char_img in characters:
                # Normalizar tamaño
                normalized = self.normalize_character_size(char_img)
                
                # Extraer características
                features = self.extract_features(normalized)
                all_features.append(features)
            
            if i % 100 == 0 and i > 0:
                print(f"Procesadas {i+1}/{len(training_images)} imágenes")
        
        if len(all_features) == 0:
            print("Error: No se encontraron características válidas después de la segmentación. Asegúrate de que las imágenes contengan caracteres detectables.")
            return None, None
        
        # 3. Clustering para etiquetado
        print("\n3. Clustering para etiquetado automático...")
        features_matrix = np.array(all_features)
        # Asegúrate de que haya suficientes muestras para el clustering
        if features_matrix.shape[0] < 2:
            print("Error: Muy pocas muestras para realizar clustering. Asegúrate de tener suficientes caracteres segmentados.")
            return None, None
        
        cluster_labels = self.cluster_characters(features_matrix)
        
        # 4. Reducción de dimensionalidad
        print("\n4. Reducción de dimensionalidad...")
        reduced_features = self.reduce_dimensionality(features_matrix)
        
        # 5. Balanceo de datos
        print("\n5. Balanceo de datos...")
        # SMOTE requiere al menos 2 muestras por clase para funcionar correctamente.
        # Si tienes muy pocas clases o clases con solo una muestra, SMOTE puede fallar.
        try:
            balanced_features, balanced_labels = self.balance_dataset(reduced_features, cluster_labels)
        except ValueError as e:
            print(f"Advertencia: No se pudo balancear el dataset con SMOTE. Esto puede deberse a muy pocas clases o clases con solo una muestra. Se continuará sin balanceo. Error: {e}")
            balanced_features, balanced_labels = reduced_features, cluster_labels

        # 6. Entrenamiento de modelos
        print("\n6. Entrenamiento de modelos...")
        # Asegúrate de tener suficientes datos para dividir en entrenamiento y prueba
        if len(np.unique(balanced_labels)) < 2:
            print("Error: No hay suficientes clases únicas en los datos para el entrenamiento del modelo. Se necesitan al menos 2.")
            return None, None
        if balanced_features.shape[0] < 2:
            print("Error: Muy pocas muestras en el dataset balanceado para el entrenamiento del modelo.")
            return None, None

        results, (X_test_scaled, y_test) = self.train_models(balanced_features, balanced_labels) # Unpack X_test_scaled, y_test
        
        # 7. Evaluación
        print("\n7. Evaluación de modelos...")
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_score'])
        best_model = results[best_model_name]['model']
        
        print(f"\nMejor modelo: {best_model_name}")
        class_names_for_eval = [str(i) for i in np.unique(y_test)] # Use y_test for class names
        evaluation = self.evaluate_model(best_model, X_test_scaled, y_test, class_names_for_eval) # Pass X_test_scaled, y_test
        
        # Plotear Matriz de Confusión
        print("\nGraficando Matriz de Confusión para el mejor modelo...")
        self.plot_confusion_matrix(evaluation['confusion_matrix'], evaluation['class_names'], 
                                   title=f"Matriz de Confusión - {best_model_name}")
        
        # Plotear Curva ROC
        print("\nGraficando Curva ROC para el mejor modelo...")
        # Pass X_test_scaled and y_test directly
        self.plot_roc_curve(best_model, X_test_scaled, y_test, evaluation['class_names'], 
                            title=f"Curva ROC - {best_model_name}")

        # Plotear Curva Precisión-Recall
        print("\nGraficando Curva Precisión-Recall para el mejor modelo...")
        # Pass X_test_scaled and y_test directly
        self.plot_precision_recall_curve(best_model, X_test_scaled, y_test, evaluation['class_names'],
                                         title=f"Curva Precisión-Recall - {best_model_name}")

        # Plotear Importancia de Características (si aplica)
        print("\nGraficando Importancia de Características para el mejor modelo (si aplica)...")
        # You might want to pass actual feature names if you have them, otherwise it will use generic ones.
        self.plot_feature_importance(best_model, title=f"Importancia de Características - {best_model_name}")

        # 8. Validación final
        print("\n8. Validación final con datos separados...")
        self._validate_with_separated_data(validation_images, best_model)
        
        # 9. Mostrar estadísticas
        print("\n=== ESTADÍSTICAS FINALES ===")
        print(f"Imágenes totales procesadas: {self.stats['total_images']}")
        print(f"Caracteres segmentados: {self.stats['segmented_chars']}")
        print(f"Caracteres únicos identificados (clusters): {self.stats['unique_chars']}")
        print(f"Características por imagen: {features_matrix.shape[1]}")
        print(f"Características después de reducción: {reduced_features.shape[1]}")
        
        return results, evaluation
    
    def _validate_with_separated_data(self, validation_images, model):
        """
        Valida el modelo con datos separados al inicio
        """
        validation_features = []
        
        for img in validation_images[:min(50, len(validation_images))]:  # Usar un máximo de 50 para validación rápida
            binary = self.binarize_image(img)
            characters = self.segment_characters(binary)
            
            for char_img in characters:
                normalized = self.normalize_character_size(char_img)
                features = self.extract_features(normalized)
                validation_features.append(features)
        
        if len(validation_features) > 0:
            validation_matrix = np.array(validation_features)
            
            # Aplicar las mismas transformaciones
            if self.pca:
                validation_reduced = self.pca.transform(validation_matrix)
            else:
                validation_reduced = validation_matrix
            
            validation_scaled = self.scaler.transform(validation_reduced)
            
            # Predicciones
            predictions = model.predict(validation_scaled)
            
            print(f"Validación con {len(validation_features)} caracteres separados")
            # Convertir las predicciones a texto si se tiene el label_encoder ajustado
            # En este caso, como los labels son de clustering, la interpretación es diferente.
            # Puedes mostrar la distribución de los clusters predichos.
            print(f"Distribución de predicciones (clusters): {Counter(predictions)}")
        else:
            print("No se pudieron procesar imágenes de validación")

# Ejemplo de uso
def main():
    # Crear una carpeta 'captcha_images' y colocar algunas imágenes de captcha dentro
    # Por ejemplo, puedes crear algunas imágenes sintéticas para probar:
    # Si no tienes imágenes, puedes descomentar la siguiente sección para generarlas.
    
    captcha_folder = "captcha"
    # Asegúrate de que la carpeta 'captcha' exista, o créala.
    if not os.path.exists(captcha_folder):
        os.makedirs(captcha_folder)
        print(f"Carpeta '{captcha_folder}' creada. Por favor, coloca algunas imágenes de captcha dentro para que el script pueda procesarlas.")
        print("Saliendo del script. Vuelve a ejecutarlo una vez que hayas añadido imágenes.")
        return
    
    # Crear instancia del sistema
    captcha_breaker = CaptchaBreaker(target_size=(32, 32))
    
    # Ejecutar pipeline completo, pasando la ruta de la carpeta
    results, evaluation_metrics = captcha_breaker.process_captcha_pipeline(folder_path=captcha_folder) # Renamed 'evaluation' to 'evaluation_metrics' for clarity
    
    # Mostrar resultados
    if results:
        print("\n=== RESULTADOS FINALES ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Precisión entrenamiento: {metrics['train_score']:.3f}")
            print(f"  Precisión prueba: {metrics['test_score']:.3f}")
            print(f"  Validación cruzada: {metrics['cv_score']:.3f} ± {metrics['cv_std']:.3f}")

if __name__ == "__main__":
    main()