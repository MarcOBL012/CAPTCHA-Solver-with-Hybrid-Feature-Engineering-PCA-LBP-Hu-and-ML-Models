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
from imblearn.over_sampling import SMOTE
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

class CaptchaBreaker:
    def __init__(self, target_size=(32, 32)):
        """
        Inicializa el objeto CaptchaBreaker.

        Input:
        - target_size (tuple[int, int]): Tamaño (alto, ancho) al que se ajustarán
          todos los caracteres segmentados.

        Atributos resultantes:
        - self.target_size (tuple[int, int]): Tamaño de destino.
        - self.label_encoder (LabelEncoder): Para codificar etiquetas de caracteres.
        - self.scaler (StandardScaler): Para normalizar características antes del entrenamiento.
        - self.pca (PCA|None): Objeto PCA (None hasta aplicar reducción de dimensionalidad).
        - self.models (dict): Para guardar modelos entrenados y sus métricas.
        - self.stats (dict): Métricas de proceso:
            * 'total_images': imágenes cargadas.
            * 'segmented_chars': caracteres detectados.
            * 'unique_chars': clusters únicos identificados.
        """
        # Guardamos el tamaño fijo al que redimensionaremos cada carácter
        self.target_size = target_size

        # Codificador de etiquetas (paso previo al entrenamiento supervisado)
        self.label_encoder = LabelEncoder()

        # Escalador estándar para normalizar las características extraídas
        self.scaler = StandardScaler()

        # Aún no hemos aplicado PCA, así que comenzamos con None
        self.pca = None

        # Diccionario donde almacenaremos cada modelo entrenado
        self.models = {}

        # Inicialización de estadísticas para seguimiento
        self.stats = {
            'total_images': 0,       # contador de imágenes cargadas
            'segmented_chars': 0,    # contador de caracteres segmentados
            'unique_chars': 0        # número de clusters detectados
        }

        # Mostrar por pantalla el estado inicial de los atributos/números
        print(f"[INIT] target_size={self.target_size}, stats={self.stats}")

    def load_images(self, folder_path):
        """
        Carga todas las imágenes .pbm desde un directorio dado.

        Input:
        - folder_path (str): Ruta al directorio que debe contener archivos .pbm

        Output:
        - images (list of ndarray): Lista de arrays de imagen cargados con OpenCV

        Transformaciones y pasos internos:
        1. Verifica que el directorio exista; si no, imprime un error y retorna lista vacía.
        2. Obtiene todos los nombres de archivo que terminan en '.pbm' (ignorando mayúsculas/minúsculas).
        3. Para cada nombre de archivo:
           a. Lee la imagen con cv2.imread.
           b. Si la lectura es exitosa (img no es None), la añade a la lista.
        4. Actualiza la estadística 'total_images' con el número de imágenes cargadas.
        5. Imprime cuántas imágenes se han cargado.
        """
        images = []

        # 1. Si la ruta no es un directorio válido, informa y retorna lista vacía
        if not os.path.isdir(folder_path):
            print(f"[LOAD_IMAGES][ERROR] El directorio '{folder_path}' no existe.")
            return images

        # 2. Filtrar sólo archivos con extensión .pbm
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pbm')]

        # 3. Intentar cargar cada archivo .pbm
        for fname in files:
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                # Aviso si algún archivo no pudo cargarse
                print(f"[LOAD_IMAGES][WARNING] No se pudo leer '{fname}'")

        # 4. Registrar cuántas imágenes se cargaron
        self.stats['total_images'] = len(images)

        # 5. Mostrar resultado de la operación
        print(f"[LOAD_IMAGES] Total imágenes cargadas: {self.stats['total_images']}")

        return images

    def binarize(self, image, threshold=127):
        """
        Convierte una imagen a un formato binario invertido.

        Input:
        - image (ndarray): Imagen en BGR o escala de grises.
        - threshold (int, opcional): Umbral para binarizar (por defecto 127).

        Output:
        - binary (ndarray): Imagen binaria invertida (píxeles del objeto a 255).

        Pasos internos:
        1. Si la imagen tiene 3 canales (BGR), la convierte a escala de grises.
        2. Aplica un umbral fijo para binarizar e invierte la salida.
        """
        # 1. Detectar si la imagen está en color (3 canales); si es así, pasar a gris
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Si ya está en escala de grises, se usa directamente

        # 2. Umbral fijo + inversión: los píxeles por encima de `threshold` pasan a 0,
        #    y los por debajo pasan a 255 (caracteres blancos sobre fondo negro)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Retornar la imagen binarizada
        return binary

    def segment(self, binary_image):
        """
        Segmenta una imagen binaria en recortes de caracteres individuales.

        Input:
        - binary_image (ndarray): Imagen en blanco y negro invertida (255 = carácter).

        Output:
        - chars (list of ndarray): Lista de sub-imágenes, cada una conteniendo un carácter detectado.

        Pasos internos:
        1. Encuentra contornos externos en la imagen binaria.
        2. Para cada contorno:
           a. Calcula el bounding box (x, y, w, h).
           b. Filtra cajas muy pequeñas (área ≤ 50 px) o con proporción demasiado estrecha/alta.
           c. Añade las cajas válidas a la lista.
        3. Ordena las cajas por coordenada x (izquierda → derecha) para mantener el orden de lectura.
        4. Recorta cada carácter de la imagen original.
        5. Actualiza la estadística 'segmented_chars' con el total de caracteres extraídos.
        """
        # 1. Detectar contornos externos
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,       # solo contornos externos
            cv2.CHAIN_APPROX_SIMPLE  # simplificación de contorno
        )

        boxes = []
        # 2. Procesar cada contorno
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            ratio = w / h if h > 0 else 0

            # 2b. Filtrar por área mínima y proporción de aspecto razonable
            if area > 50 and 0.2 < ratio < 2.0:
                boxes.append((x, y, w, h))
            else:
                print(f"[SEGMENT][FILTER] Contorno descartado: área={area}, ratio={ratio:.2f}")

        # 3. Ordenar cajas de izquierda a derecha
        boxes.sort(key=lambda b: b[0])

        # 4. Recortar caracteres de la imagen binaria original
        chars = [binary_image[y:y+h, x:x+w] for (x, y, w, h) in boxes]

        # 5. Actualizar estadística de segmento
        self.stats['segmented_chars'] = len(chars)
        #print(f"[SEGMENT] Caracteres segmentados: {self.stats['segmented_chars']}")

        return chars

    def normalize(self, char_img):
        """
        Redimensiona y centra un carácter segmentado a un tamaño fijo.

        Input:
        - char_img (ndarray): Imagen binaria del carácter (variable de tamaño).

        Output:
        - frame (ndarray): Imagen de tamaño self.target_size, con el carácter escalado y centrado.

        Pasos internos:
        1. Obtener dimensiones originales (h0, w0) de la imagen.
        2. Calcular factor de escala para ajustar proporcionalmente al 80% del espacio disponible.
        3. Redimensionar el carácter con cv2.resize (si new_h/new_w > 0).
        4. Crear un 'frame' negro de tamaño target_size y 'pegar' el carácter escalado centrado.
        """
        # 1. Dimensiones originales del carácter
        h0, w0 = char_img.shape
        H, W = self.target_size

        # 2. Cálculo de la escala (manteniendo proporción y margen del 20%)
        scale = min(H / h0, W / w0) * 0.8
        new_h, new_w = int(h0 * scale), int(w0 * scale)

        # 3. Redimensionar, o crear vacío si la escala es nula o negativa
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(char_img, (new_w, new_h))
        else:
            resized = np.zeros(self.target_size, dtype=np.uint8)

        # 4. Generar frame en negro y centrar la imagen escalada
        frame = np.zeros(self.target_size, dtype=np.uint8)
        dh, dw = (H - new_h) // 2, (W - new_w) // 2
        frame[dh:dh + new_h, dw:dw + new_w] = resized

        return frame

    def extract(self, char_img):
        """
        Extrae un vector de características de una imagen de carácter normalizada.

        Input:
        - char_img (ndarray): Imagen binaria de tamaño fijo (self.target_size).

        Output:
        - features (ndarray): Vector 1D que concatena:
            * pix: valores de píxel normalizados (flatten / 255).
            * hist: histograma de intensidad de 16 bins.
            * mu: 7 momentos invariantes de Hu.
            * lbp: histograma LBP de 32 bins.

        Pasos internos:
        1. Aplanar píxeles y normalizar a [0,1].
        2. Calcular histograma de 16 bins con cv2.calcHist.
        3. Obtener momentos de Hu a partir de cv2.moments.
        4. Calcular descriptor LBP llamando a self._lbp.
        5. Concatenar todas las características en un solo vector.
        """
        # 1. Valores de píxel normalizados
        pix = char_img.flatten() / 255.0

        # 2. Histograma de intensidades (16 bins)
        hist = cv2.calcHist([char_img], [0], None, [16], [0, 256]).flatten()

        # 3. Momentos invariantes de Hu (7 valores)
        mu = cv2.HuMoments(cv2.moments(char_img)).flatten()

        # 4. Descriptor LBP (32 bins)
        lbp = self._lbp(char_img)

        # 5. Concatenar todas las características
        features = np.hstack([pix, hist, mu, lbp])
        return features

    def _lbp(self, img):
        """
        Calcula el descriptor Local Binary Pattern (LBP) de una imagen binaria.

        Input:
        - img (ndarray): Imagen binaria de tamaño fijo.

        Output:
        - hist_norm (ndarray): Histograma LBP de 32 bins, normalizado.

        Pasos internos:
        1. Iterar sobre cada píxel interior (excluyendo bordes) y comparar con sus 8 vecinos.
        2. Generar un código binario de 8 bits: cada bit indica si el vecino es ≥ al píxel central.
        3. Acumular todos los códigos en la lista `feats`.
        4. Calcular histograma de los códigos (`np.histogram` con 32 bins).
        5. Normalizar el histograma dividiendo por la suma más una pequeña constante.
        """
        h, w = img.shape
        feats = []

        # 1. Recorrer píxeles interiores (sin bordes)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                c = img[i, j]
                code = 0
                # Lista de valores de los 8 píxeles vecinos
                neigh = [
                    img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                    img[i,   j+1],
                    img[i+1, j+1], img[i+1, j], img[i+1, j-1],
                    img[i,   j-1]
                ]
                # 2. Construir código: bit k = 1 si vecino ≥ píxel central
                for k, n in enumerate(neigh):
                    code |= (int(n >= c) << k)
                feats.append(code)

        # 4. Histograma de los códigos LBP (32 bins)
        hist, _ = np.histogram(feats, bins=32, range=[0, 256])
        # 5. Normalización y retorno
        hist_norm = hist / (hist.sum() + 1e-7)
        return hist_norm

    def cluster(self, X, n_clusters=None):
        if n_clusters is None:
            n_clusters=2
        k=KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
        labs=k.fit_predict(X)
        self.stats['unique_chars']=n_clusters
        return labs

    def cluster(self, X, n_clusters=None):
        """
        Agrupa las muestras en clusters usando K-Means y asigna etiquetas.

        Input:
        - X (ndarray): Matriz de características (muestras × dimensiones).
        - n_clusters (int|None): Número de clusters. Si es None, se usa 2 por defecto.

        Output:
        - labs (ndarray): Etiquetas de cluster para cada muestra.

        Pasos internos:
        1. Definir número de clusters (por defecto 2).
        2. Instanciar KMeans con semilla fija y 10 reinicios.
        3. Ajustar KMeans y predecir etiquetas de cluster.
        4. Actualizar la estadística 'unique_chars' con n_clusters.
        5. Retornar las etiquetas.
        """
        # 1. Valor por defecto si no se especifica
        if n_clusters is None:
            n_clusters = 2

        # 2. Configurar KMeans
        k = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10  # número de inicializaciones diferentes para robustez
        )

        # 3. Ajustar modelo y obtener etiquetas de cluster
        labs = k.fit_predict(X)

        # 4. Guardar cuántos clusters hemos usado
        self.stats['unique_chars'] = n_clusters

        # 5. Devolver etiquetas para cada muestra
        print(f"[CLUSTER] Usados {n_clusters} clusters, etiquetas generadas para {len(labs)} muestras")
        return labs

    def split_data(self, X, y, test_size=0.15):
        """
        Separa los datos en conjuntos de entrenamiento y verificación.

        Input:
        - X (ndarray): Matriz de características.
        - y (ndarray): Vector de etiquetas.
        - test_size (float, opcional): Fracción para el conjunto de verificación.

        Output:
        - X_tr (ndarray): Características de entrenamiento.
        - X_vf (ndarray): Características de verificación.
        - y_tr (ndarray): Etiquetas de entrenamiento.
        - y_vf (ndarray): Etiquetas de verificación.

        Pasos internos:
        1. Llamar a train_test_split con estratificación para mantener proporción de clases.
        2. Imprimir tamaños de cada partición.
        """
        # 1. División estratificada para conservar proporción de clases
        X_tr, X_vf, y_tr, y_vf = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # 2. Mostrar por pantalla cuántos ejemplos hay en cada conjunto
        print(f"[SPLIT_DATA] train={len(y_tr)}, verif={len(y_vf)}")

        return X_tr, X_vf, y_tr, y_vf

    def balance(self, X, y):
        """
        Aplica oversampling para corregir desbalance de clases usando SMOTE.

        Input:
        - X (ndarray): Matriz de características de entrenamiento.
        - y (ndarray): Vector de etiquetas de entrenamiento.

        Output:
        - X_res (ndarray): Nuevas características balanceadas.
        - y_res (ndarray): Nuevas etiquetas balanceadas.

        Pasos internos:
        1. Instanciar SMOTE con semilla fija para reproducibilidad.
        2. Ajustar y re-muestrear X, y para generar nuevas muestras sintéticas.
        3. Retornar los datos balanceados.
        """
        # 1. Crear SMOTE con random_state para resultados reproducibles
        sm = SMOTE(random_state=42)

        # 2. Generar nuevos ejemplos para balancear las clases
        X_res, y_res = sm.fit_resample(X, y)
        print(f"[BALANCE] Ejemplos antes={len(y)}, después={len(y_res)}")

        # 3. Devolver características y etiquetas balanceadas
        return X_res, y_res

    def train(self, X, y):
        """
        Entrena múltiples clasificadores tras escalar los datos.

        Input:
        - X (ndarray): Matriz de características balanceadas.
        - y (ndarray): Vector de etiquetas balanceadas.

        Output:
        - res (dict): Diccionario con, para cada modelo:
            * 'model': objeto del clasificador entrenado.
            * 'train_acc': precisión en el conjunto de entrenamiento.
            * 'cv_mean': precisión media en validación cruzada 5‑fold.

        Pasos internos:
        1. Estandarizar X con StandardScaler.
        2. Definir tres modelos: RandomForest (RF), MLP, Decision Tree (DT).
        3. Para cada modelo:
           a. Ajustar (`fit`) sobre los datos escalados.
           b. Calcular precisión en entrenamiento (`score`).
           c. Validar por cross_val_score (5 folds).
           d. Guardar métricas en `res` y mostrar un `print`.
        4. Almacenar `res` en `self.models`.
        """
        # 1. Escalar características
        Xs = self.scaler.fit_transform(X)

        # 2. Definir modelos a entrenar
        models = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            'DT': DecisionTreeClassifier(random_state=42)
        }

        res = {}
        # 3. Entrenar y evaluar cada modelo
        for name, model in models.items():
            # a. Ajustar modelo
            model.fit(Xs, y)
            # b. Precisión en training set
            train_acc = model.score(Xs, y)
            # c. Validación cruzada 5‑fold
            cv_scores = cross_val_score(model, Xs, y, cv=5)
            # d. Guardar resultados y mostrar
            res[name] = {
                'model': model,
                'train_acc': train_acc,
                'cv_mean': cv_scores.mean()
            }
            print(f"[TRAIN] {name}: train {train_acc:.3f}, cv {cv_scores.mean():.3f}")

        # 4. Actualizar modelos entrenados
        self.models = res
        return res

    def evaluate(self, model, X_test, y_test):
        """
        Evalúa un modelo entrenado en datos de verificación.

        Input:
        - model (estimator): Clasificador entrenado.
        - X_test (ndarray): Características del conjunto de verificación.
        - y_test (ndarray): Etiquetas verdaderas de verificación.

        Output:
        - acc (float): Precisión (accuracy) del modelo en el conjunto de verificación.

        Pasos internos:
        1. Escalar X_test usando el mismo scaler que en entrenamiento.
        2. Calcular precisión con model.score.
        3. Imprimir la precisión obtenida.
        4. Retornar el valor de precisión.
        """
        # 1. Transformar datos de verificación con StandardScaler entrenado
        Xt = self.scaler.transform(X_test)

        # 2. Obtener precisión en verificación
        acc = model.score(Xt, y_test)

        # 3. Mostrar resultado
        print(f"[EVALUATE] Precisión en verificación: {acc:.3f}")

        # 4. Retornar la métrica de accuracy
        return acc

    def dimensionality(self, X, n_components=0.95):
        """Aplica PCA para reducir dimensiones y guarda el PCA en `self.pca`."""
        self.pca = PCA(n_components=n_components)
        Xr = self.pca.fit_transform(X)
        print(f"[PCA] {X.shape[1]}→{Xr.shape[1]}, var exp:{self.pca.explained_variance_ratio_.sum():.3f}")
        return Xr

    def process(self, folder_path):
        """
        Ejecuta el pipeline completo: desde cargar imágenes hasta evaluar modelos.

        Input:
        - folder_path (str): Ruta al directorio que contiene imágenes CAPTCHA (.pbm).

        Output:
        - results (dict): Métricas de entrenamiento y validación para cada modelo.

        Pasos internos:
        1. Carga todas las imágenes del directorio usando load_images.
        2. Para cada imagen:
           a. Binarizarla (binarize).
           b. Segmentar caracteres (segment).
           c. Para cada carácter:
              i. Normalizar tamaño (normalize).
              ii. Extraer características (extract).
              iii. Añadir a la lista all_feat.
        3. Construir matriz X con todas las características.
        4. Agrupar con K-Means para generar etiquetas y actualizar stats (cluster).
        5. Reducir dimensionalidad con PCA (dimensionality).
        6. Dividir datos en entrenamiento y verificación (split_data).
        7. Balancear set de entrenamiento con SMOTE (balance).
        8. Entrenar modelos y obtener métricas (train).
        9. Evaluar cada modelo en el set de verificación (evaluate) e imprimir resultados.
        10. Retornar diccionario con resultados.
        """
        # 1. Carga de imágenes
        imgs = self.load_images(folder_path)

        # 2–3. Preprocesamiento y extracción de todas las características
        all_feat = []
        for img in imgs:
            b = self.binarize(img)
            chars = self.segment(b)
            for c in chars:
                n = self.normalize(c)
                f = self.extract(n)
                all_feat.append(f)

        # 4. Matriz de características
        X = np.array(all_feat)

        # 5. Clustering para etiqueta inicial
        y = self.cluster(X)

        # 6. PCA para reducir dimensiones
        Xr = self.dimensionality(X) # guardar como atributo para visualización
        self.Xr = Xr          
        self.y = y 

        # 7. División en train/validation
        Xt, Xv, yt, yv = self.split_data(Xr, y)# guardar set de verificacíon
        self.X_verif = Xv                       
        self.y_verif = yv

        # 8. Balanceo del conjunto de entrenamiento
        Xb, yb = self.balance(Xt, yt)

        # 9. Entrenamiento de modelos
        results = self.train(Xb, yb)

        # 10. Evaluación en verificación
        print("[PROCESS] Evaluación en set de verificación:")
        for name, info in results.items():
            model = info['model']
            self.evaluate(model, Xv, yv)

        return results

def plot_pca(X_pca, y_labels):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_labels, cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("Visualización PCA (2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.show()

def plot_confusion(model, X_test, y_test, scaler):
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()

    
    

if __name__ == "__main__":
    # 1. Instanciar la clase CaptchaBreaker
    cb = CaptchaBreaker()
    # 2. Ejecutar el pipeline completo sobre la carpeta 'captcha'
    results = cb.process('captcha')

    # Graficar matriz de confusión para el modelo RF
    best_model = results['RF']['model']
    plot_confusion(best_model, cb.X_verif, cb.y_verif, cb.scaler)

    # 3. Mostrar resumen de resultados finales
    print(f"[MAIN] Resultados finales: {results}")