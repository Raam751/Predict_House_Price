# 🏠 House Price Prediction

A Streamlit web app that predicts house prices using a Linear Regression model trained on the famous Housing dataset.

## Features

- Predicts house prices based on 12 features: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hot water heating, air conditioning, parking, preferred area, and furnishing status.
- Displays model evaluation metrics (R² Score & RMSE) in the sidebar.
- Shows feature coefficients and an interactive bar chart after prediction.

## Demo

1. Adjust the house features using the input fields.
2. Click **Predict Price** to see the predicted price.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The [Housing dataset](Housing.csv) contains 545 records with the following columns:

| Feature | Description |
|---------|-------------|
| price | House price (target) |
| area | Area in sqft |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of stories |
| mainroad | Connected to main road (yes/no) |
| guestroom | Has guest room (yes/no) |
| basement | Has basement (yes/no) |
| hotwaterheating | Has hot water heating (yes/no) |
| airconditioning | Has air conditioning (yes/no) |
| parking | Number of parking spaces |
| prefarea | In preferred area (yes/no) |
| furnishingstatus | furnished / semi-furnished / unfurnished |
