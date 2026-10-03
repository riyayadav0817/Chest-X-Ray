# Chest X-ray Disease Classification using DenseNet121

A deep learning project for **multi-label classification of 14 thoracic diseases from chest X-ray images** using the NIH ChestX-ray14 dataset and a DenseNet121 convolutional neural network.

> **Note:** This project is intended for educational and research purposes and is not a clinical diagnostic system.

---

## 📌 Project Overview

Chest X-ray images can contain multiple abnormalities simultaneously. This project builds a multi-label deep learning model that predicts the presence of **14 different thoracic findings** from a single chest X-ray.

The model uses **DenseNet121** and produces one prediction score for each disease.

### Diseases

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax

---

## 🗂️ Dataset

The project uses the **NIH ChestX-ray14 dataset**, containing more than 100,000 chest X-ray images with disease labels.

The dataset includes:

* 112,120 total X-ray images
* 14 disease labels
* Patient-level metadata
* Official training/validation and test lists

### Dataset Split

| Split          |      Images |
| -------------- | ----------: |
| Training       |      77,988 |
| Validation     |       8,536 |
| Official Test  |      25,596 |
| **Total Used** | **112,120** |

The official test set was kept completely separate from training and model selection.

---

## 🧠 Model Architecture

### DenseNet121

The model uses **DenseNet121** as the convolutional neural network architecture.

The final classifier was modified to output 14 values:

```text
Input Image
     ↓
Resize 224 × 224
     ↓
DenseNet121
     ↓
Fully Connected Layer
     ↓
14 Disease Scores
     ↓
Sigmoid
     ↓
Disease Probabilities
```

### Important

The model was **trained from scratch** because pretrained ImageNet weights could not be downloaded in the Kaggle environment.

---

## ⚙️ Data Preprocessing

Each image was:

* Converted to RGB
* Resized to `224 × 224`
* Converted to a PyTorch tensor
* Normalized using:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

A patient-level split was used for the training/validation data to reduce patient leakage.

---

## ⚖️ Handling Class Imbalance

The dataset contains significant differences in disease frequency.

To address this, the model uses:

```python
BCEWithLogitsLoss(pos_weight=pos_weight)
```

Positive class weights were calculated from the training set and capped at `50.0` to prevent extremely rare classes from dominating the loss.

---

## 🔧 Training Configuration

| Parameter       | Value                 |
| --------------- | --------------------- |
| Model           | DenseNet121           |
| Initialization  | Random / From Scratch |
| Input Size      | 224 × 224             |
| Batch Size      | 32                    |
| Epochs          | 5                     |
| Optimizer       | AdamW                 |
| Learning Rate   | 0.0001                |
| Weight Decay    | 0.0001                |
| Loss Function   | BCEWithLogitsLoss     |
| Class Weighting | Capped at 50          |
| GPU             | NVIDIA Tesla T4       |

---

## 📊 Model Performance

### Validation

Best validation performance was obtained at **Epoch 5**.

| Metric                 |      Score |
| ---------------------- | ---------: |
| Validation Macro AUROC | **0.7860** |

### Official Test Set

The final model was evaluated on all **25,596 official test images**.

| Metric           |      Score |
| ---------------- | ---------: |
| Test Macro AUROC | **0.7472** |
| Macro Precision  | **0.1890** |
| Macro Recall     | **0.4021** |
| Macro F1-score   | **0.2429** |

The Macro AUROC was calculated across all 14 disease labels.

---

## 📈 Disease-wise Test AUROC

| Disease            |  AUROC |
| ------------------ | -----: |
| Atelectasis        | 0.7195 |
| Cardiomegaly       | 0.8635 |
| Consolidation      | 0.7104 |
| Edema              | 0.8192 |
| Effusion           | 0.7825 |
| Emphysema          | 0.7594 |
| Fibrosis           | 0.7483 |
| Hernia             | 0.8598 |
| Infiltration       | 0.6718 |
| Mass               | 0.7176 |
| Nodule             | 0.6414 |
| Pleural Thickening | 0.7212 |
| Pneumonia          | 0.6595 |
| Pneumothorax       | 0.7874 |

**Overall Test Macro AUROC: 0.7472**

---

## 🎯 Threshold Optimization

Because this is a highly imbalanced multi-label problem, using a fixed probability threshold of `0.5` produced high recall but relatively low precision.

Disease-specific thresholds were therefore selected using the **validation set only** by maximizing F1-score.

The selected thresholds were then applied once to the untouched official test set.

This resulted in:

```text
Macro Precision = 0.1890
Macro Recall    = 0.4021
Macro F1        = 0.2429
```

The test AUROC remains:

```text
0.7472
```

Threshold optimization does not change AUROC because AUROC evaluates the ranking of prediction scores rather than one fixed classification threshold.

---

## 📊 Visualizations

The project includes visualizations for:

* ROC curves for all 14 diseases
* Disease-wise AUROC
* Precision / Recall / F1-score
* Training loss
* Final model performance

---

## 💻 Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* PIL
* CUDA
* Kaggle GPU

---

## 📁 Project Structure

```text
Chest-X-Ray/
│
├── README.md
├── chest_xray_training.ipynb
│
├── visualizations/
│   ├── roc_curves.png
│   ├── disease_auroc.png
│   ├── precision_recall_f1.png
│   ├── training_loss.png
│   └── final_performance.png
│
└── models/
    └── best_densenet121.pth
```

> Large model checkpoints may be better stored using Git LFS, a Kaggle output, or another artifact-storage service rather than committing them directly to a normal Git repository.

---

## 🚀 Future Improvements

Possible improvements include:

* Transfer learning with ImageNet-pretrained DenseNet121
* More training epochs
* Learning-rate scheduling
* Data augmentation
* Better handling of rare diseases
* Disease-specific threshold optimization
* Hyperparameter tuning
* Grad-CAM visualization
* Calibration of predicted probabilities
* Comparison with other architectures such as EfficientNet and ResNet
* External validation on another chest X-ray dataset

---

## ⚠️ Limitations

This project has several limitations:

1. The DenseNet121 model was trained from scratch rather than using pretrained ImageNet weights.
2. Only 5 training epochs were used.
3. The dataset contains substantial class imbalance.
4. Some diseases have very few positive examples.
5. Performance varies considerably across disease categories.
6. The model has not been clinically validated.
7. Predictions should not be used for medical diagnosis or treatment decisions.

---

## 📌 Conclusion

This project demonstrates a complete deep learning pipeline for multi-label chest X-ray classification using DenseNet121.

The final model achieved:

**Validation Macro AUROC: 0.7860**

**Official Test Macro AUROC: 0.7472**

The project also demonstrates patient-level dataset splitting, class-imbalance handling, threshold optimization, ROC analysis, and disease-wise evaluation.

---

## 👩‍💻 Author

**Rashmi Chaudhary**


**Skills:** Python • Machine Learning • Computer Vision • Deep Learning • AWS
