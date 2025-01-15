# Developing a Classifier for Movie Ratings Using the IMDB Dataset

## Overview

This project aims to develop a machine learning classifier to predict movie ratings based on data from the IMDB dataset. The goal is to bin continuous IMDB scores into discrete categories for classification, utilizing a variety of features including actor and director popularity, movie earnings, and textual embeddings.

The work involves key phases:

1. **Preprocessing and Exploratory Data Analysis (EDA)** - Cleaning and validating datasets, and exploring the data structure and distributions.
2. **Feature Engineering and Dimensionality Reduction** - Enhancing dataset quality using techniques like PCA and PLS regression.
3. **Feature Selection and Model Training** - Optimizing features for improved model performance.
4. **Model Evaluation** - Testing various algorithms and selecting the best-performing model based on robust metrics.

---

## Key Results

The best-performing model for this task is **Gradient Boosted Decision Trees (XGBoost)** using a set of features obtained from a build up feature selection over a standardised data set. This model demonstrated superior handling of the categorical target feature and the class imbalance present in the dataset. It provided the highest accuracy and balanced performance across all IMDB score bins. 

### Model Performance (Key Metrics)

- **Accuracy:** ~71%
- **F1-Score (Weighted):** ~0.69
- **Balanced Accuracy:** Highlighted robust performance on minority classes.

---

## Features

### Inputs

- **Numerical Features:** Includes critic reviews, social media likes, movie duration, and earnings.
- **Categorical Features:** Actor names, director names, genres, and content ratings.
- **Textual Features:** Embeddings derived from keywords and genres.

### Target

The target variable is a binned version of the IMDB score, divided into five discrete categories.

---

## Technologies and Methods Used

1. **Data Processing:**
   - Handling missing values and duplicates.
   - Scaling and encoding features for model compatibility.

2. **Dimensionality Reduction:**
   - Principal Component Analysis (PCA) and Partial Least Squares (PLS) regression.

3. **Machine Learning Models:**
   - Logistic Regression
   - Support Vector Machines (SVM)
   - Random Forest
   - Neural Networks
   - **Gradient Boosted Decision Trees (XGBoost)** (Best model)

4. **Evaluation Metrics:**
   - Accuracy
   - F1-Score
   - Balanced Accuracy

---

## Usage

1. **Prerequisites:**
   - Python 3.8+
   - Libraries: scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn.

2. **Running the Project:**
  - Jupyter notebook contains everything you will need, run seqeuentially.

---

## Acknowledgments

This project was inspired by the wealth of data available through the IMDB and aimed to provide insights into user ratings. Special thanks to contributors and the open-source community for their tools and resources.
