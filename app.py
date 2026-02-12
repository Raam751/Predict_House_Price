import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px


def train_model():
    df = pd.read_csv('Housing.csv')

    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, X.columns, r2, rmse


def main():
    st.title('Simple Linear Regression Model for House Price Prediction')
    st.write('This app predicts house prices based on various features using a linear regression model.')

    model, feature_columns, r2, rmse = train_model()

    st.sidebar.header('Model Metrics')
    st.sidebar.write(f'R² Score: {r2:.4f}')
    st.sidebar.write(f'RMSE: {rmse:,.2f}')

    area = st.number_input('Area (sqft)', min_value=500, max_value=20000, value=5000)
    bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
    stories = st.number_input('Number of Stories', min_value=1, max_value=5, value=2)
    mainroad = st.selectbox('Main Road', ['yes', 'no'])
    guestroom = st.selectbox('Guest Room', ['yes', 'no'])
    basement = st.selectbox('Basement', ['yes', 'no'])
    hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
    airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
    parking = st.number_input('Parking Spaces', min_value=0, max_value=5, value=1)
    prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
    furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

    if st.button('Predict Price'):
        input_data = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': 1 if mainroad == 'yes' else 0,
            'guestroom': 1 if guestroom == 'yes' else 0,
            'basement': 1 if basement == 'yes' else 0,
            'hotwaterheating': 1 if hotwaterheating == 'yes' else 0,
            'airconditioning': 1 if airconditioning == 'yes' else 0,
            'parking': parking,
            'prefarea': 1 if prefarea == 'yes' else 0,
            'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
            'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0,
        }])
        input_data = input_data[feature_columns]

        predicted_price = model.predict(input_data)[0]
        st.success(f'Predicted House Price: ₹{predicted_price:,.2f}')

        coeff_df = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False)
        st.subheader('Model Coefficients')
        st.dataframe(coeff_df)

        fig = px.bar(coeff_df, x='Feature', y='Coefficient', title='Features vs Coefficient')
        st.plotly_chart(fig)


if __name__ == '__main__':
    main()
