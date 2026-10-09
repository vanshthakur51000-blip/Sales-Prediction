# Sales Prediction Using Python

## Project Overview

This project uses Machine Learning to predict sales based on advertising expenditure on TV, Radio, and Newspaper.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

## Dataset

The `Advertising.csv` dataset contains 200 records with advertising budgets and sales figures.

## Algorithm

Linear Regression is used to train the model and predict sales.

## Model Performance

* **MAE:** 1.46
* **MSE:** 3.17
* **R² Score:** 0.90

## How to Run

Install the required libraries:

```bash
python -m pip install pandas scikit-learn matplotlib joblib streamlit
```

Run the application:

```bash
streamlit run app.py
```

## Features

* Enter advertising budgets for TV, Radio, and Newspaper.
* Predict sales using the trained model.
* View results in an interactive web application.

## Conclusion

This project demonstrates how Machine Learning can analyze advertising data and predict sales to support marketing decisions.
