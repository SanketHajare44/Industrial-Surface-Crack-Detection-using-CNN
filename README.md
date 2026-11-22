# Industrial Surface Crack Detection using CNN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-CNN-red?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Computer_Vision-OpenCV-green" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Project Status">
</p>

A deep learning-based industrial defect detection system that uses a Convolutional Neural Network (CNN) to classify surface images as **cracked (positive)** or **non-cracked (negative)**. The project uses TensorFlow and Keras with image preprocessing, augmentation, model training, evaluation, and single-image prediction to support automated visual quality inspection.

---

## 📌 Overview

Surface cracks can compromise the safety, reliability, and quality of industrial materials and infrastructure. Manual inspection can be time-consuming, so this project explores an image-classification approach to identify cracks automatically.

The model learns visual patterns from labeled images and predicts whether a given surface image contains a crack.

## ✨ Features

- **Binary image classification:** Classifies images into `Crack` and `NoCrack`.
- **Dataset preparation:** Organizes the original positive and negative image sets into training, validation, and testing subsets.
- **Train/validation/test split:** Uses a 70% / 15% / 15% split.
- **Image preprocessing:** Resizes images to 128 × 128 and rescales pixel values.
- **Data augmentation:** Applies rotation, zoom, horizontal/vertical shifting, and horizontal flipping to training images.
- **CNN architecture:** Uses multiple convolution and pooling layers to learn image features.
- **Regularization:** Includes Batch Normalization and Dropout.
- **Training callbacks:** Uses Early Stopping, Model Checkpointing, and ReduceLROnPlateau.
- **Performance evaluation:** Reports test loss, test accuracy, confusion matrix, and classification report.
- **Visualization:** Plots training and validation accuracy and loss.
- **Single-image prediction:** Loads an image and predicts whether a crack is detected.

## 🧰 Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** Keras image utilities
- **Data Handling:** NumPy
- **Visualization:** Matplotlib
- **Evaluation:** Scikit-learn

## 📂 Repository Structure

```text
Industrial-Surface-Crack-Detection-using-CNN/
│
├── CrackDataset/
│   ├── Positive/        # Images containing cracks
│   └── Negative/        # Images without cracks
│
├── industrial_surface_crack_detection.py
└── README.md
```

> When you run the script, it creates a `Processed_CrackDataset/` directory containing the training, validation, and testing splits. The trained model files are also generated locally.

## 🗂️ Dataset

The dataset is arranged into two classes:

| Class      | Description                      |
| ---------- | -------------------------------- |
| `Positive` | Surface images containing cracks |
| `Negative` | Surface images without cracks    |

The script automatically creates the processed dataset with the following structure:

```text
Processed_CrackDataset/
├── train/
│   ├── Crack/
│   └── NoCrack/
├── validation/
│   ├── Crack/
│   └── NoCrack/
└── test/
    ├── Crack/
    └── NoCrack/
```

### Dataset Split

| Subset     | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

The training set is used to learn model parameters, the validation set is used to monitor training, and the test set is used for final evaluation.

## 🧠 CNN Architecture

The model is built using Keras Sequential and contains convolutional feature-extraction blocks followed by dense classification layers.

| Layer                  | Configuration                             |
| ---------------------- | ----------------------------------------- |
| Input                  | 128 × 128 RGB image                       |
| Convolution Block 1    | 32 filters, 3 × 3, ReLU                   |
| Convolution Block 2    | 64 filters, 3 × 3, ReLU                   |
| Convolution Block 3    | 128 filters, 3 × 3, ReLU                  |
| Convolution Block 4    | 256 filters, 3 × 3, ReLU                  |
| Each convolution block | Batch Normalization + Max Pooling         |
| Flatten                | Converts extracted features into a vector |
| Dense Layer 1          | 256 units, ReLU                           |
| Dropout                | 0.5                                       |
| Dense Layer 2          | 128 units, ReLU                           |
| Dropout                | 0.3                                       |
| Output Layer           | 1 unit, Sigmoid                           |

**Compilation configuration:**

- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy

## ⚙️ Training Configuration

| Parameter         | Value               |
| ----------------- | ------------------- |
| Image size        | 128 × 128           |
| Batch size        | 32                  |
| Maximum epochs    | 15                  |
| Random seed       | 42                  |
| Optimizer         | Adam                |
| Loss function     | Binary Crossentropy |
| Output activation | Sigmoid             |

### Data Augmentation

Augmentation is applied to the training data to introduce variation and help the model generalize:

- Rotation up to 15°
- Zoom up to 20%
- Width shift up to 10%
- Height shift up to 10%
- Horizontal flipping

Validation and test images are rescaled without augmentation.

### Callbacks

- **EarlyStopping:** Monitors validation loss and restores the best weights when training stops.
- **ModelCheckpoint:** Saves the model with the best validation accuracy.
- **ReduceLROnPlateau:** Reduces the learning rate when validation loss stops improving.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/SanketHajare44/Industrial-Surface-Crack-Detection-using-CNN.git
cd Industrial-Surface-Crack-Detection-using-CNN
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

### 4. Verify Dataset Folders

Make sure the original dataset folders exist:

```text
CrackDataset/
├── Positive/
└── Negative/
```

### 5. Run the Project

```bash
python industrial_surface_crack_detection.py
```

The script prepares the dataset splits, trains the CNN, displays training visualizations, evaluates the model on test data, saves the models, and performs a sample single-image prediction (if a test image is available).

## 📊 Evaluation

The model is evaluated on the test subset, which is not used for training. The script outputs:

- Test Loss
- Test Accuracy
- Confusion Matrix
- Classification Report (precision, recall, F1-score)

It also plots:

- Training vs. Validation Accuracy
- Training vs. Validation Loss

> Add your measured test accuracy and screenshots/plots here after running the project. Do not report a metric unless it comes from your actual run.

## 💾 Saved Models

The script saves the trained model files:

- `Best_Crack_Detection_Model.keras` — checkpoint saved when validation accuracy improves.
- `Final_Crack_Detection_Model.keras` — model saved at the end of training.

## 🔍 Single Image Prediction

The project includes a `predict_single_image()` function that preprocesses an individual image, runs it through the trained CNN, and displays the predicted class.

Predicted labels:

- `Crack Detected`
- `No Crack`

The function uses the same image size and normalization as the model's training pipeline.

## 🔮 Future Improvements

- Add a user-friendly interface for uploading images and viewing predictions.
- Experiment with transfer learning architectures such as MobileNetV2 or EfficientNet.
- Add precision-recall and ROC-AUC visualizations.
- Improve generalization using a larger and more diverse dataset.
- Explore object localization or segmentation to highlight crack regions.
- Deploy the model as an API or web application.

## 🤝 Contributions

Suggestions, issues, and improvements are welcome. Feel free to fork the repository and experiment with alternative architectures, preprocessing methods, or evaluation strategies.

## 👤 Author

**Sanket Sadashiv Hajare**

- GitHub: [SanketHajare44](https://github.com/SanketHajare44)
- LinkedIn: [Sanket Hajare](https://www.linkedin.com/in/sankethajare/)

---

⭐ If you find this project interesting, consider starring the repository.
