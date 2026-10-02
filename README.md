# Wine Classification Machine Learning - 

## Project Description

This project implements a Machine Learning classification application for classifying wines into three different classes based on their chemical properties.

The dataset contains chemical analysis results of wines from three different cultivars. The model uses the available wine features to predict the corresponding wine class.

## Dataset

The dataset used in this project is:

`WinePredictor.csv`

The dataset contains 13 input features and one target variable.

### Features

- `Alcohol`
- `Malic acid`
- `Ash`
- `Alcalinity of ash`
- `Magnesium`
- `Total phenols`
- `Flavanoids`
- `Nonflavanoid phenols`
- `Proanthocyanins`
- `Color intensity`
- `Hue`
- `OD280/OD315 of diluted wines`
- `Proline`

### Target

`Class`

The wine is classified into:

- Class 1
- Class 2
- Class 3

The dataset description and these 13 features are specified in the Assignment 41 document. :contentReference[oaicite:0]{index=0}

## Machine Learning Model

The project uses:

- Decision Tree Classifier
- Train-Test Split
- Accuracy Score
- Confusion Matrix
- Classification Report

## Machine Learning Workflow

The application follows these steps:

```text
Get Data
    ↓
Clean, Prepare and Manipulate Data
    ↓
Train Data
    ↓
Test Data
    ↓
Calculate Accuracy
