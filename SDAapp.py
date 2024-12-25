import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import smtplib
from email.mime.text import MIMEText

# Load Data
def load_data():
    uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        return df
    return None

# Data Visualization
def visualize_data(df):
    st.subheader("Sales Overview")
    if 'Sales' in df.columns and 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        daily_sales = df.resample('D', on='Order Date').sum(numeric_only=True)
        
        st.line_chart(daily_sales['Sales'])

        st.subheader("Top Products by Sales")
        top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
        st.bar_chart(top_products)

        st.subheader("Sales by City")
        city_sales = df.groupby('City')['Sales'].sum()
        st.bar_chart(city_sales)
    else:
        st.error("CSV must contain 'Order Date' and 'Sales' columns.")

# Sales Forecasting
# Forecasting Results
def forecast_sales(df):
    if 'Sales' in df.columns and 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
        monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
        monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])
        
        # Prepare Data for Forecasting
        X = monthly_sales.index.values.reshape(-1, 1)
        y = monthly_sales['Sales'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.subheader("Forecasting Results")
        plt.figure(figsize=(10, 5))
        
        # Corrected plotting
        plt.plot(monthly_sales['Month'], y, label='Actual Sales')
        
        # Corrected line for predicted sales
        plt.plot(monthly_sales['Month'].iloc[X_test.flatten()], y_pred, label='Predicted Sales', linestyle='--')
        
        plt.legend()
        st.pyplot(plt)

# Email Notification
def send_email_notification(body, to_email):
    from_email = 'youremail@example.com'
    password = 'yourpassword'
    
    msg = MIMEText(body)
    msg['Subject'] = 'Monthly Sales Report'
    msg['From'] = from_email
    msg['To'] = to_email
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            st.success("Email Sent Successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Streamlit App
st.title("Sales Data Analysis and Forecasting")
data = load_data()

if data is not None:
    visualize_data(data)
    forecast_sales(data)
    
    st.subheader("Send Monthly Report")
    email = st.text_input("Enter recipient email:")
    if st.button("Send Report"):
        send_email_notification("Monthly sales report is ready.", email)