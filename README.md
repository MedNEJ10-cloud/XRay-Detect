# XRay-Detect: Deep Learning for Pulmonary Disease Classification

Welcome to the repository for **XRay-Detect**! This project focuses on leveraging computer vision to accurately classify chest X-ray images into three categories: Covid-19, Normal, and Pneumonia.

Instead of relying on heavy, complex ensembles, this project fine-tunes a single, robust Convolutional Neural Network (CNN) architecture to achieve high clinical reliability while keeping inference computationally efficient.

## Dataset & Preprocessing
The data pipeline is handled carefully to ensure the model learns generalized features:
*   **Data split:** The dataset is divided into 80% for training and 20% for testing using a stratified split to maintain class balance.
*   **Preprocessing:** Every image is converted to RGB format and resized to a standard 224x224 pixel. 
*   **Augmentation:** To prevent overfitting and build robust feature extraction, we apply random rotations (up to 30 degrees), width/height shifts (20%), shearing, zooming, and horizontal flips via Keras `ImageDataGenerator`.
*   Processed data is saved efficiently into NumPy arrays (`trainX.npy`, `trainY.npy`, `testX.npy`, `testY.npy`) for faster loading during experiments.

## Model Architecture
The core of this diagnostic tool relies on Transfer Learning utilizing **InceptionV3**:
*   **Base Model:** InceptionV3 initialized with pre-trained `imagenet` weights, expecting an input tensor of shape `(224, 224, 3)`.
*   **Custom Head:** The default top layers were replaced with a `GlobalAveragePooling2D` layer, followed by a `Dense` layer of 250 units (ReLU activation), a `Dropout` of 0.5 for regularization, and a final 3-unit `Softmax` layer for multi-class prediction.
*   **Fine-tuning Strategy:** The first 102 layers of the base model are frozen, allowing the subsequent deeper layers and the new custom head to be specifically fine-tuned on our medical dataset.

## Training Details
*   **Optimizer:** `Adam` optimizer is used to minimize categorical cross-entropy loss.
*   **Callbacks:** We implemented a dynamic learning process using `ReduceLROnPlateau` (halving the learning rate when validation loss stagnates), `EarlyStopping` (patience=7) to halt training if the model stops improving, and `ModelCheckpoint` to guarantee we save only the best iteration of the weights (`X-rays_best_model.h5`).

## Results & Performance
The model demonstrates exceptional capability on the unseen test set, achieving an overall global accuracy of **98.33%** (with a loss of just 0.0626).
Detailed classification report metrics on the test set:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Covid-19** | 0.98 | 0.97 | 0.97 | 200 |
| **Normal** | 0.98 | 0.98 | 0.98 | 200 |
| **Pneumonia** | 0.99 | 1.00 | 1.00 | 200 |

### Evaluation Visuals
*The evaluation notebook generates a Confusion Matrix (`matrice_X-rayx-Model.png`) and ROC Curves (`roc_curve_model.png`) to visualize True Positive Rates against False Positive Rates across all three classes*. 



Le système s'appuie sur le modèle de Deep Learning **InceptionV3** en utilisant des techniques de transfer learning et de data augmentation. Le modèle final, sauvegardé après optimisation du taux d'apprentissage, atteint une précision globale de **98,33%** sur les données de test, démontrant d'excellentes performances (F1-score de 1.00 pour la pneumonie) et prouvant son potentiel en tant qu'outil d'aide au diagnostic radiologique.
