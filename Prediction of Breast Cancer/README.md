# Breast Cancer Predictionn

This repository hosts a Python script dedicated to building and training a neural network model designed to classify breast cancer diagnoses, utilizing features extracted from medical images. The dataset essential for this project is stored within a CSV file, 'data.csv', which contains pertinent information concerning breast cancer cases.

**Prerequisites**

**Before executing the code, make sure the following libraries are installed:**

- Pandas
- NumPy
- TensorFlow
- Scikit-learn
- Seaborn
- Matplotlib

**You can easily install these libraries with the following command:**
1. install these libraries on your local machine.**
   ```bash
      pip install pandas numpy tensorflow scikit-learn seaborn matplotlib

2. Usage

    Begin by cloning the repository to your local machine:

   ```bash
    git clone https://github.com/sheikhshaheernaeem/prediction-of-Breast-Cancer.git


3. Next, run the Python script:
 
   ```bash
      python prediction-of-Breast-Cancer.py


**The script conducts the following key operations:**

- Reads the dataset from 'data.csv'.
- Preprocesses the data, including addressing missing values and encoding categorical variables.
- Segments the dataset into training and testing sets.
- Constructs a deep neural network model for classifying breast cancer diagnoses.
- Trains the model with the training data and evaluates it using validation data.
- Generates predictions employing the trained model on the testing data.
- Computes a confusion matrix and visually represents it using a heatmap.


**File Description**

1. breast_cancer_classification.py: The primary Python script housing the code for data preprocessing, model development, training, evaluation, and visualization.
2. data.csv: The dataset file in CSV format encompassing breast cancer diagnostic data and related features.
3. README.md: This README file supplies comprehensive information about the project.

**Results**
The script empowers the training of a neural network model aimed at classifying breast cancer diagnoses based on the provided dataset. The heatmap of the confusion matrix offers valuable insights into the model's performance on the testing data.

**Acknowledgments**
The dataset employed in this project is sourced from a reputable origin (please provide source details if applicable).
The neural network architecture is crafted utilizing TensorFlow and Keras.
Data visualization is enhanced through the utilization of Seaborn and Matplotlib.
