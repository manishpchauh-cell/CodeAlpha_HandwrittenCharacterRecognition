# Handwritten Character Recognition using CNN

## CodeAlpha Machine Learning Internship - Task 3

## Project Overview

This project is developed as part of the CodeAlpha Machine Learning Internship.

The objective of this project is to identify handwritten digits using image processing and deep learning. The project uses the MNIST dataset and a Convolutional Neural Network (CNN) model to classify handwritten digits from 0 to 9.

---

## Objective

To build a deep learning model that can recognize handwritten digits accurately from image data.

---

## Dataset

The project uses the MNIST handwritten digit dataset.

The dataset contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 x 28 pixels
- Classes: digits 0 to 9

---

## Model Used

The project uses a Convolutional Neural Network (CNN).

CNN layers used:

- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Dropout
- Softmax Output Layer

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── handwritten_character_recognition.ipynb
├── handwritten_character_recognition.py
├── README.md
├── requirements.txt
├── Project_Report.pdf
│
├── models/
│   └── handwritten_cnn_model.h5
│
└── images/
    ├── sample_digits.png
    ├── training_accuracy.png
    ├── confusion_matrix.png
    └── prediction_samples.png
```

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-github-username/CodeAlpha_HandwrittenCharacterRecognition.git
```

### Step 2: Open the Project Folder

```bash
cd CodeAlpha_HandwrittenCharacterRecognition
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Notebook

Open:

```text
handwritten_character_recognition.ipynb
```

Run all cells in Google Colab or Jupyter Notebook.

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Loss
- Confusion Matrix
- Classification Report
- Sample Predictions

---

## Output

The project generates:

- Sample digit images
- Training accuracy graph
- Confusion matrix
- Sample prediction results
- Trained CNN model file

---

## Future Scope

- Extend to EMNIST alphabets dataset
- Recognize handwritten words
- Use CRNN for sequence recognition
- Deploy using Streamlit or Flask
- Improve accuracy using data augmentation

---

## Author

**Sandesh Solagi**

Machine Learning Intern  
CodeAlpha Internship Program

---

## License

This project is developed for educational and internship purposes.
