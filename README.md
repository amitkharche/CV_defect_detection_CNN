
---

# Apple Defect Classifier: A Computer Vision Project for Smart Quality Control

## Overview

This project implements a deep learning-based computer vision pipeline to classify apple images as **defective** or **non-defective**. It is tailored for quality control use cases in agriculture, packaging, and manufacturing environments.

By training a Convolutional Neural Network (CNN), the system can identify surface-level defects such as bruises, rot, and blemishes from images, enabling faster and more consistent visual inspections.

---

## Model Highlights

- Model: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input: 64x64 RGB images
- Output: Binary classification (`defective`, `non-defective`)
- Preprocessing: Resizing and normalization (pixel values scaled to [0, 1])
- Evaluation Metrics: Accuracy, precision, recall, F1-score (via `classification_report`)

---

## Project Structure

```

CV\_defect\_detection\_CNN/
├── data/
│   ├── images/                    # Apple images (clean and defective)
│   └── image\_labels.csv          # CSV with image paths and string labels
├── model/
│   ├── cnn\_defect\_model.h5       # Trained Keras model
│   └── class\_map.pkl             # Label map for Streamlit prediction
├── model\_training.py             # Training script for CNN
├── app.py                        # Streamlit web app for image prediction
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

````

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/amitkharche/CV_defect_detection_CNN.git
cd CV_defect_detection_CNN
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Place all your apple images inside the `data/images/` folder. Then, create a CSV file at `data/image_labels.csv` with the following format:

```csv
image_path,label
data/images/apple_clean_1.jpg,non-defective
data/images/apple_defective_1.jpg,defective
...
```

Ensure that all image paths are relative to the project root and labels are either `defective` or `non-defective`.

---

## Training the Model

To train the CNN on your dataset:

```bash
python model_training.py
```

This will:

* Train the model on your labeled images
* Save the model to `model/cnn_defect_model.h5`
* Save the class index map to `model/class_map.pkl`

---

## Running the Streamlit App

To classify apple images using the trained model:

```bash
streamlit run app.py
```

The app allows you to:

* Upload a JPG or PNG image
* Get an instant classification (`defective` / `non-defective`)
* View prediction confidence and guidance messages

---

## Example Output

* Classification: **Defective**
* Confidence: `89.23%`
* Suggestion: High confidence – recommend automated rejection.

---

## License

This project is licensed under the MIT License.

---

## Author

**Amit Kharche**
[GitHub](https://github.com/amitkharche)
[LinkedIn](https://www.linkedin.com/in/amitkharche)
[Medium](https://medium.com/@amitkharche)

---
