# Accident Severity Detection

A Flask web application that uses machine learning models to predict the severity of road accidents based on various input features.

## Features

- Upload accident dataset (Excel format)
- View and preprocess the dataset
- Train multiple machine learning models:
  - Random Forest Classifier
  - XGBoost Classifier
  - Support Vector Classifier (SVC)
  - Logistic Regression
  - CatBoost Classifier
- Predict accident severity (Fatal, Serious, or Slight) based on input parameters

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VishalReddyYalla/Accident-Severity-Detection.git
   cd Accident-Severity-Detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`.

## Usage

1. **Load Data** – Upload an Excel (`.xlsx`) file containing accident records.
2. **View Data** – Inspect the first 100 rows of the uploaded dataset.
3. **Preprocess** – Clean the data, encode categorical variables, apply SMOTE oversampling, and split into train/test sets.
4. **Train Model** – Select and train a machine learning algorithm.
5. **Predict** – Enter feature values to predict the severity of an accident.

## Input Features

| Feature | Description |
|---|---|
| Lighting Conditions | Light level at the time of the accident |
| Road Surface | Condition of the road surface |
| Weather Conditions | Weather at the time of the accident |
| Casualty Class | Class of casualty (driver, passenger, etc.) |
| Sex of Casualty | Sex of the casualty |
| Type of Vehicle | Type of vehicle involved |
| Speed | Speed of the vehicle |
| Number of Vehicles | Number of vehicles involved |
| Age of Casualty | Age of the casualty |
| Time (24hr) | Time of the accident in 24-hour format |
| 1st Road Class | Classification of the road |

## Output Classes

- **Fatal** – Life-threatening injuries
- **Serious** – Serious injuries requiring hospital treatment
- **Slight** – Minor injuries
