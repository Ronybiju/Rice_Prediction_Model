# 🌾 Rice Classification Using CNN

A deep learning-based image classification project to identify different types of rice grains using Convolutional Neural Networks (CNN). Built in Python using TensorFlow/Keras and trained on a labeled image dataset, this project implements regularization and augmentation techniques to improve model performance and reduce overfitting.

---

## 📁 Dataset

The dataset contains images of various rice types stored in class-specific folders:

```
/rice_dataset/
  ├── Basmati/
  ├── Jasmine/
  ├── SonaMasoori/
  └── ...
```

- Images are resized to 224×224 pixels.
- Dataset loaded using `ImageDataGenerator` with validation split and augmentation.

---

## 🛠️ Tech Stack

- Python
- TensorFlow & Keras
- NumPy, Matplotlib
- Google Colab

---

## 🧠 Model Architecture

- **Input:** 224×224×3 image
- **Layers:**
  - Conv2D → ReLU → MaxPooling2D → Dropout
  - Conv2D → ReLU → MaxPooling2D → Dropout
  - Flatten → Dense (L2 regularized) → Dropout
  - Output: Dense with Softmax activation

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Callbacks Used:** EarlyStopping (to avoid overfitting)

---

## 🧪 Model Performance

| Epoch | Training Accuracy | Validation Accuracy |
|-------|-------------------|---------------------|
| 9     | 64.28%            | **83.33%**          |
| 12    | 75.30%            | 72.00%              |

> ✅ Best Validation Accuracy: **83.33%**

---

## 🔍 Features

- ✅ Deep CNN Model
- ✅ Dropout + L2 Regularization
- ✅ Real-Time Data Augmentation
- ✅ EarlyStopping for Generalization
- ✅ Supports custom image predictions

---

## 🚀 How to Use

### 🔗 Clone the Repository
```bash
git clone https://github.com/yourusername/rice-classification-cnn.git
cd rice-classification-cnn
```

### 💻 Run the Notebook (Colab Recommended)
- Open `rice_classification.ipynb` in Google Colab.
- Mount your Google Drive and upload `rice_dataset`.
- Follow cell-by-cell instructions to train and test the model.

### 🖼️ Predict a Custom Image
```python
from tensorflow.keras.preprocessing import image

img = image.load_img("your_image.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_label = list(class_indices.keys())[np.argmax(prediction)]
print("Predicted Rice Type:", predicted_label)
```

---

## 📌 Future Scope

- Deploy as Web App using Streamlit or Flask
- Convert model to TensorFlow Lite for edge devices
- Add Grad-CAM heatmaps for explainability

---
