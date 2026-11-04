# 🚦 Indian Road Accident Severity Prediction Using Machine Learning

## Problem Statement

Road accidents are a major public safety concern in India. Every year, thousands of accidents occur due to factors such as weather conditions, road characteristics, traffic density, visibility issues, and driver-related factors. The severity of these accidents can vary significantly, ranging from minor incidents to fatal collisions.

Identifying patterns associated with accident severity is important for improving road safety, supporting emergency response systems, and assisting transportation authorities in understanding accident risks.

However, manually analyzing large volumes of accident data is time-consuming and difficult. This creates a need for a data-driven approach that can analyze accident-related information and predict the likely severity category of an accident.

---

## Project Aim

The main aim of this project is to develop a Machine Learning-based system capable of predicting the severity of road accidents using historical accident data from India.

The model classifies accidents into the following categories:

- Minor
- Major
- Fatal

By learning patterns from historical accident records, the model can estimate the likely severity level based on various accident-related factors.

---

## Proposed Solution

To address the problem, a Machine Learning pipeline was developed using Python and Scikit-Learn.

The pipeline performs:

1. Data Cleaning
2. Missing Value Treatment
3. Feature Selection
4. Feature Encoding
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Severity Prediction

The final system allows users to provide accident-related details and receive a predicted accident severity category.

---

## How This Project Works

The system takes accident-related information as input, including:

- Weather Conditions
- Road Type
- Visibility
- Traffic Density
- Temperature
- Number of Vehicles Involved
- Number of Casualties
- Peak Hour Information
- Traffic Signal Availability
- Weekend Indicator
- Festival Information

The trained Machine Learning model processes these inputs and predicts whether the accident belongs to:

- Minor Severity
- Major Severity
- Fatal Severity

---

## Expected Benefits

This project can contribute to:

### Road Safety Analysis

Understanding the conditions under which severe accidents occur.

### Accident Pattern Identification

Finding trends related to weather, traffic, and road conditions.

### Decision Support

Providing data-driven insights for future road safety studies.

### Educational Purpose

Demonstrating the use of Machine Learning techniques in solving real-world transportation and safety problems.

---

# 📊 Dataset Description

The project uses an Indian Road Accident Dataset containing accident-related information collected from various road and traffic conditions.

### Input Features

| Feature | Description |
|----------|------------|
| hour | Hour of accident occurrence |
| day_of_week | Day on which accident occurred |
| road_type | Type of road |
| weather | Weather condition during accident |
| visibility | Visibility level |
| traffic_density | Traffic conditions |
| lanes | Number of lanes |
| temperature | Environmental temperature |
| vehicles_involved | Number of vehicles involved |
| casualties | Number of casualties |
| traffic_signal | Traffic signal availability |
| is_peak_hour | Peak hour indicator |
| is_weekend | Weekend indicator |
| festival | Festival information |
| risk_score | Calculated risk score |

### Target Variable

**Accident Severity**

Categories:

- Minor
- Major
- Fatal

---

# 🔍 Exploratory Data Analysis

Several exploratory analyses were performed using:

- Pandas
- NumPy
- Matplotlib
- Seaborn

Visualizations include:

- Accident Severity Distribution
- Temperature Distribution
- Hour-wise Accident Analysis
- Weekend vs Weekday Analysis
- Temperature vs Risk Score Correlation
- Traffic Density vs Accident Severity

These visualizations helped identify trends and relationships within the dataset.

---

# ⚙️ Data Preprocessing

The following preprocessing operations were performed:

### Missing Value Handling

Missing values in the Festival column were replaced with:

```python
No Festival
```

### Date Conversion

The date column was converted into datetime format.

### Binary Feature Processing

The following fields were converted into numerical format:

- is_weekend
- traffic_signal
- is_peak_hour

### One-Hot Encoding

Categorical features were transformed using:

```python
OneHotEncoder()
```

### Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

### Train-Test Split

The dataset was divided into:

- Training Data = 70%
- Testing Data = 30%

using stratified sampling.

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

### Logistic Regression

A baseline classification model.

### Decision Tree Classifier

Used for rule-based classification.

### Random Forest Classifier

Used to improve predictive performance using ensemble learning.

### Voting Classifier

The final ensemble model combining multiple classifiers.

---

# 📈 Evaluation Metrics

Model performance was evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

# 🖥️ Streamlit Application

A Streamlit web application was developed to make the system interactive and user-friendly.

### Features

✅ Clean Interface

✅ User Input Form

✅ Severity Prediction

✅ Severity Explanation

✅ Real-Time Results

The application allows users to enter accident-related information and instantly receive the predicted accident severity.

---

# 📂 Project Structure

```text
INDIAN_ROAD_ACCIDENT_SEVERITY_PREDICTION/

│
├── data/
│   └── indian_roads_accidents_dataset.csv
│
├── notebooks/
│   └── road_accident_analysis.ipynb
│
├── model/
│   └── road_accident_model.pkl
│
├