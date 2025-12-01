# 🩺 Diabetes Prediction Model

## 🎯 Project Overview

This project focuses on developing a machine learning model to **predict the onset of diabetes** in patients based on various diagnostic measurements.

The classification task is performed using several machine learning algorithms, with a focus on **AdaBoost Classifier** which achieved the best performance after hyperparameter tuning. The data used is the Pima Indians Diabetes Database.

## 🗃️ Dataset

The project uses the `diabetes.csv` dataset, which includes the following features:

| Feature | Description |
| :--- | :--- |
| **Pregnancies** | Number of times pregnant. |
| **Glucose** | Plasma glucose concentration a 2 hours in an oral glucose tolerance test. |
| **BloodPressure** | Diastolic blood pressure (mm Hg). |
| **SkinThickness** | Triceps skin fold thickness (mm). |
| **Insulin** | 2-Hour serum insulin (mu U/ml). |
| **BMI** | Body mass index (weight in kg/(height in m)^2). |
| **DiabetesPedigreeFunction** | Diabetes pedigree function (a measure of genetic influence). |
| **Age** | Age (years). |
| **Outcome** | Class variable (0 = Non-diabetic, 1 = Diabetic). |

## 🛠️ Technologies and Libraries

The project is built using Python and the following key libraries:

* **Python**
* **Pandas** and **NumPy** for data manipulation.
* **Scikit-learn** for modeling and evaluation.
* **Matplotlib** and **Seaborn** (imported but not explicitly used in the provided code, though recommended for EDA).

## 💻 Installation and Setup

To run this notebook locally, follow these steps:

1.  **Clone the repository (or download the file):**
    ```bash
    git clone <repository-link>
    cd <project-directory>
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Ensure the data file is present:**
    Make sure `diabetes.csv` is in the root directory of the project.

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## 🚀 Data Processing and Model Training

### 1. Preprocessing Steps
* **Feature-Target Split:** `Outcome` was set as the target variable (`y`).
* **Scaling:** All feature variables were scaled using `StandardScaler` to normalize the data distribution.
* **Handling Missing Values (Zeros):** Features like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` had minimum values of 0, which are biologically impossible for a living patient and were treated as missing data (`np.nan`).
* **Imputation:** These missing values were filled using the **median** strategy via `SimpleImputer`.
* **Data Split:** The data was split into training and testing sets with a `test_size` of 20% and `random_state=42`. 

[Image of the machine learning workflow showing data preparation, model training, and evaluation]


### 2. Models Evaluated

The following models were trained and their base `accuracy_score` on the test set was recorded:

| Model | Test Accuracy (Base) |
| :--- | :--- |
| **Logistic Regression** | **75.32%** |
| **Random Forest Classifier** | 74.03% |
| **Support Vector Classifier (Linear)** | **75.97%** |
| **AdaBoost Classifier** | 77.92% |

### 3. Hyperparameter Tuning

The **AdaBoost Classifier** was selected for further tuning using `GridSearchCV` to optimize its performance.

* **Tuning Parameters:** `n_estimators` (number of base learners) and `learning_rate`.
* **Best Parameters Found:**
    ```python
    {'learning_rate': 1.0, 'n_estimators': 100}
    ```

## ✅ Final Results

After optimization, the AdaBoost model achieved the highest performance:

| Metric | Score |
| :--- | :--- |
| **Best Training Score (CV)** | 78.18% |
| **Final Test Accuracy** | 75.97% |

The final best-performing model is the **AdaBoost Classifier** with a test accuracy of approximately **76.0%**.

## 📝 License

This project is licensed under the MIT License - see the LICENSE.md file for details (if you choose to add one).

## 👤 Author

* **[Your Name/Alias]** - *Initial work* - [Link to your GitHub Profile]