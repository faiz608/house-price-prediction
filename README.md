# 🏠 House Price Prediction Using Machine Learning

A machine learning project that analyzes housing data and predicts house sale prices using Python, exploratory data analysis, data preprocessing, regression models, and CatBoost.

## 📌 Project Overview

The objective of this project is to develop a machine learning model that predicts residential house sale prices based on property characteristics, including overall quality, living area, lot size, garage features, basement area, construction year, and other housing attributes.

The project follows a complete machine learning workflow, from understanding and analyzing the dataset to training regression models, comparing their performance, and evaluating predictions on unseen test data.

## 🎯 Project Objectives

* Explore and understand the housing dataset.
* Perform exploratory data analysis (EDA).
* Identify missing values and analyze data quality.
* Investigate relationships between housing features and sale prices.
* Prepare numerical and categorical features for machine learning.
* Train and compare multiple regression models.
* Perform hyperparameter tuning using GridSearchCV.
* Evaluate model performance using RMSE and R².
* Analyze feature importance to understand influential predictors.
* Generate house price predictions using the trained model.

## 📊 Dataset Information

The notebook loads the dataset from a CSV file named `data.csv`.

The dataset contains **1,460 rows and 81 columns**, including the target variable and the property identifier.

| Property          | Description |
| ----------------- | ----------- |
| Dataset file      | `data.csv`  |
| Number of records | 1,460       |
| Total columns     | 81          |
| Target variable   | `SalePrice` |
| Identifier column | `Id`        |
| Prediction task   | Regression  |

The dataset includes numerical and categorical information describing residential properties.

### Examples of Important Features

| Feature        | Description                         |
| -------------- | ----------------------------------- |
| `OverallQual`  | Overall material and finish quality |
| `GrLivArea`    | Above-ground living area            |
| `LotArea`      | Lot size                            |
| `GarageCars`   | Garage capacity in cars             |
| `GarageArea`   | Garage area                         |
| `TotalBsmtSF`  | Total basement area                 |
| `YearBuilt`    | Original construction year          |
| `YearRemodAdd` | Remodeling year                     |
| `FullBath`     | Number of full bathrooms            |
| `TotRmsAbvGrd` | Total rooms above ground            |

The model uses `SalePrice` as the target variable. The `Id` column is excluded from the input features.

**Note:** The notebook does not specify the original dataset download URL. Add the dataset source here once it has been verified.

## 🛠️ Technologies and Libraries

The project uses the following Python libraries:

* **Python** — Programming language.
* **Pandas** — Data manipulation and analysis.
* **NumPy** — Numerical computations.
* **Matplotlib** — Data visualization.
* **Seaborn** — Statistical visualization.
* **Scikit-learn** — Data preprocessing, regression models, pipelines, hyperparameter tuning, and evaluation.
* **CatBoost** — Gradient boosting regression.
* **Jupyter Notebook** — Interactive development and analysis.

## 🔍 Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to understand the dataset and identify patterns that may influence house prices.

The notebook includes:

1. **Dataset inspection:** Examining the dataset shape, column types, and summary statistics.
2. **Sale price distribution:** Visualizing the distribution of house prices using histograms.
3. **Log-transformed distribution:** Examining the distribution of `log1p(SalePrice)`.
4. **Missing value analysis:** Identifying columns with missing observations and visualizing their distribution.
5. **Correlation analysis:** Examining numerical features correlated with `SalePrice`.
6. **Living area analysis:** Investigating the relationship between `GrLivArea` and `SalePrice`.
7. **Overall quality analysis:** Comparing sale prices across different overall quality ratings.

The correlation analysis identified `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, and `TotalBsmtSF` among the numerical features most strongly associated with sale price in the dataset.

Correlation indicates an association, not necessarily a causal relationship.

## 🧹 Data Preprocessing

The dataset contains both numerical and categorical features, along with missing values.

The project uses a Scikit-learn preprocessing pipeline to prepare the input data.

### Numerical Features

Missing numerical values are handled using median imputation.

### Categorical Features

Missing categorical values are handled using most-frequent-value imputation.

Categorical variables are then transformed using One-Hot Encoding, with unknown categories ignored during transformation.

### Feature and Target Separation

* **Input features (`X`):** All columns except `SalePrice` and `Id`.
* **Target (`y`):** `SalePrice`.

### Train-Test Split

The dataset is divided into training and testing subsets:

* Training set: 80%
* Testing set: 20%
* Random state: 42

The same split configuration is used to make the model comparisons reproducible.

## 🤖 Machine Learning Models

The notebook evaluates four regression approaches.

### 1. Ridge Regression

Ridge Regression provides a regularized linear regression baseline.

Its regularization parameter, `alpha`, is selected using GridSearchCV.

### 2. Random Forest Regressor

Random Forest combines predictions from multiple decision trees to model nonlinear relationships between property features and sale prices.

The notebook searches different values for:

* Number of estimators
* Maximum tree depth
* Minimum samples required to split a node

### 3. Gradient Boosting Regressor

Gradient Boosting builds an ensemble of decision trees sequentially to improve predictive performance.

The notebook tunes:

* Number of estimators
* Learning rate
* Maximum tree depth

### 4. CatBoost Regressor

CatBoost is used as the final regression model.

The model is configured with:

* Iterations: 2,000
* Learning rate: 0.05
* Depth: 8
* Loss function: RMSE
* Random seed: 42

The final CatBoost model is trained using the preprocessing pipeline and evaluated on the held-out test set.

## ⚙️ Hyperparameter Tuning

GridSearchCV is used to tune Ridge Regression, Random Forest, and Gradient Boosting.

The search uses three-fold cross-validation and negative root mean squared error as the scoring metric.

The best parameters are selected based on cross-validation performance.

CatBoost is trained separately using the specified configuration above; the notebook does not apply GridSearchCV to that model.

## 📈 Model Evaluation

The models are evaluated using two metrics.

### Root Mean Squared Error (RMSE)

RMSE measures the typical magnitude of prediction errors while giving greater weight to larger errors.

A lower RMSE indicates better predictive performance.

### R² Score

R² measures how much of the variation in the target variable is explained by the model on the evaluation dataset.

A score closer to 1 generally indicates a better fit.

### Results

The following results were recorded in the notebook.

| Model                       |      RMSE | R² Score |
| --------------------------- | --------: | -------: |
| Ridge Regression            | 34,571.97 |   0.8442 |
| Random Forest Regressor     | 28,393.01 |   0.8949 |
| Gradient Boosting Regressor | 27,356.81 |   0.9024 |
| CatBoost Regressor          | 26,876.17 |   0.9058 |

### Best Result

Based on the recorded test-set results, CatBoost achieved the lowest RMSE and highest R² score among the four evaluated models.

* **RMSE:** Approximately 26,876.17
* **R² Score:** Approximately 0.9058

An R² score of approximately 0.906 indicates that the model explains about 90.6% of the variation in sale prices in this test set.

RMSE is expressed in the same units as the target variable. The interpretation of the numerical error therefore depends on the currency and scale represented by the dataset.

These results describe the particular train-test split used in the notebook and do not guarantee the same performance on other datasets.

## 📊 Prediction Visualization

The notebook generates an actual-versus-predicted scatter plot to compare predicted sale prices with observed values.

A reference line represents perfect predictions. Points closer to this line indicate smaller prediction errors.

This visualization helps identify how closely the model's predictions match the actual sale prices.

## 🔑 Feature Importance

Feature importance is extracted from the trained CatBoost model and visualized for the top 25 features.

The leading features in the recorded results include:

* `OverallQual`
* `GrLivArea`
* `BsmtFinSF1`
* `TotalBsmtSF`
* `LotArea`
* `GarageCars`
* `1stFlrSF`
* `YearBuilt`

These results suggest that property quality, living area, basement characteristics, lot size, garage capacity, and construction year are important predictors for this fitted model.

Feature importance describes the model's use of information for prediction. It does not establish that a feature causes a change in house prices.

## 💻 Project Structure

The repository is organized as follows:

```text
house-price-prediction/
│
├── house.ipynb
├── data.csv
├── README.md
└── requirements.txt
```

* `house.ipynb` — Jupyter Notebook containing the analysis, preprocessing, model training, evaluation, and prediction.
* `data.csv` — Dataset required to execute the notebook.
* `README.md` — Project documentation.
* `requirements.txt` — Python dependencies required to run the project.

The dataset and requirements file should be added to the repository if you want others to reproduce the analysis. The structure above represents the recommended layout.

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/faiz608/house-price-prediction.git
```

### Step 2: Navigate to the Project Directory

```bash
cd house-price-prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Step 5: Add the Dataset

Make sure `data.csv` is present in the same directory as `house.ipynb`.

The notebook loads the dataset using:

```python
df = pd.read_csv("data.csv")
```

If the file is missing or has a different name or location, update the file path in the notebook.

### Step 6: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `house.ipynb` in the Jupyter interface.

### Step 7: Run the Notebook

Execute the notebook cells sequentially, from importing libraries and loading the dataset to training the models and evaluating their predictions.

## ⚠️ Important Notes

* The dataset file `data.csv` is required to run the notebook.
* The notebook contains an installation command for CatBoost. In a local environment, install dependencies before running the notebook.
* The reported metrics are based on the saved notebook outputs and may differ when the notebook is executed again.
* The dataset's original source and licensing should be verified before redistributing it.
* The model is a learning and analysis project, not a validated professional property valuation service.
* No standalone web application or deployment interface is included in the notebook.

## 🔮 Future Improvements

Potential improvements include:

* Add a verified dataset source and license information.
* Compare additional regression algorithms.
* Explore more systematic hyperparameter optimization.
* Examine residuals and prediction errors in greater detail.
* Evaluate performance across multiple train-test splits or cross-validation folds.
* Save the trained model for reuse.
* Build an interactive house price prediction interface using Streamlit or another suitable framework.
* Improve project reproducibility by documenting the Python environment and dataset version.

## 📝 Conclusion

This project demonstrates an end-to-end machine learning workflow for residential house price prediction.

It covers exploratory data analysis, missing value handling, categorical feature encoding, regression model comparison, hyperparameter tuning, model evaluation, and feature importance analysis.

Among the evaluated models, CatBoost achieved the strongest recorded test-set performance, with an RMSE of approximately 26,876.17 and an R² score of approximately 0.9058.

The project provides practical experience with Python, Scikit-learn, CatBoost, and machine learning techniques for structured tabular data.

## 👨‍💻 Author

**Faizan Ali**

GitHub: [@faiz608](https://github.com/faiz608)

Project Repository: [house-price-prediction](https://github.com/faiz608/house-price-prediction)
