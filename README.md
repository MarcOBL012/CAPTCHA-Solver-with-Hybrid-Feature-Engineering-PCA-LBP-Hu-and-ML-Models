# CAPTCHA-Solver-with-Hybrid-Feature-Engineering-PCA-LBP-Hu-and-ML-Models


## 📋 Overview

**CaptchaBreaker** is an automated system designed to process, segment, and recognize characters from textual CAPTCHA images. While CAPTCHAs are designed to distinguish humans from bots, advances in Computer Vision and Machine Learning allow us to solve them with high precision.

This project implements a full pipeline: from data acquisition using Selenium to character classification using Supervised Learning models (Random Forest, MLP, Decision Trees). A key feature of this project is the use of **Unsupervised Clustering (K-Means)** to generate labels for training data when real labels are unavailable.


## 🚀 Features

* **Automated Data Acquisition:** Scrapes CAPTCHA images using Selenium.
* **Preprocessing:** Grayscale conversion and adaptive binarization.
* **Segmentation:** Contour detection to isolate individual characters.
* **Feature Extraction:** Extracts robust features including:
    * Normalized Pixel Values
    * Intensity Histograms (16 bins)
    * Hu Moments (7 invariant moments)
    * Local Binary Patterns (LBP - 32 bins)
* **Auto-Labeling:** Uses **K-Means Clustering** to group similar characters and assign surrogate labels.
* **Dimensionality Reduction:** Uses **PCA** (Principal Component Analysis) to retain 95% variance.
* **Data Balancing:** Implements **SMOTE** to handle class imbalances.
* **Multi-Model Classification:** Trains and evaluates:
    * Random Forest
    * Multi-Layer Perceptron (Neural Network)
    * Decision Tree

## 🛠️ Tech Stack

* **Python 3.x**
* **OpenCV** (`cv2`): Image processing and segmentation.
* **Scikit-Learn**: Machine learning models, PCA, K-Means, metrics.
* **Imbalanced-Learn**: SMOTE for oversampling.
* **Selenium**: Data scraping/collection.
* **Matplotlib & Seaborn**: Data visualization and confusion matrices.
* **Pandas & NumPy**: Data manipulation.

## ⚙️ Methodology Pipeline

The system follows a strict processing pipeline:

1.  **Acquisition:** Downloads 10,000+ `.pbm` or `.png` CAPTCHA images.
2.  **Preprocessing:** Images are binarized (black characters on white background) and inverted.
3.  **Segmentation:** Characters are cropped using bounding boxes derived from contours.
4.  **Normalization:** Crops are resized to a fixed target size (e.g., 32x32).
5.  **Feature Extraction:** A 1D feature vector is created for each character.
6.  **Clustering & Labeling:** Characters are grouped into $K$ clusters to generate training labels.
7.  **Dimensionality Reduction:** PCA reduces the feature space (e.g., from 1079 dimensions to ~4).
8.  **Training:** Supervised models are trained on the balanced, reduced dataset.
9.  **Evaluation:** Performance is measured using Accuracy, Confusion Matrices, and Cross-Validation.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MarcOBL012/CAPTCHA-Solver-with-Hybrid-Feature-Engineering-PCA-LBP-Hu-and-ML-Models.git
    cd CaptchaBreaker
    ```

2.  **Install dependencies:**
    You can install the required libraries using pip:
    ```bash
    python -m pip install numpy opencv-python matplotlib scikit-learn imbalanced-learn seaborn pandas selenium
    ```

##  ▶️ Usage

1.  **Prepare Data:**
    Create a folder named `captcha` in the root directory and place your CAPTCHA images (e.g., `.pbm`, `.png`) inside it.
    *(Note: The code includes a scraper if you need to download them first).*

2.  **Run the Main Script:**
    Execute the Python script to start the processing pipeline.

    ```bash
    python app.py
    ```

3.  **View Results:**
    The script will output the training progress, cross-validation scores, and finally display:
    * Classification Report (Precision, Recall, F1-Score).
    * Confusion Matrix Plot.
    * ROC and Precision-Recall Curves.

## 📬 Contact
If you use or extend this project, please add a note in the README or contact:

Marco Obispo — marco.obispo.l@uni.pe
