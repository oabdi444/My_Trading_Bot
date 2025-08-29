import streamlit as st
import pandas as pd

# Load the CSV file (must be in the same folder as this script)
@st.cache_data
def load_data():
    df = pd.read_csv('all_stocks_5yr.csv')
    return df

df = load_data()

# UI: Select a stock ticker
stock_list = df['Name'].unique()
selected_stock = st.selectbox('Select Stock', sorted(stock_list))

# Filter for the chosen stock
df_stock = df[df['Name'] == selected_stock].copy()
df_stock['date'] = pd.to_datetime(df_stock['date'])
df_stock = df_stock.set_index('date')
df_stock = df_stock.drop(columns=['Name'])

st.title(f"{selected_stock} Stock Dashboard")

# Show raw data
st.write("### Raw Data")
st.dataframe(df_stock.head(20))

# Summary statistics
st.write("### Summary Statistics")
st.dataframe(df_stock.describe())

# Line chart: Close price
st.write("### Close Price Over Time")
st.line_chart(df_stock['close'])

# Bar chart: Volume
st.write("### Volume Over Time")
st.bar_chart(df_stock['volume'])

# Optional: date range selector
st.write("### Select Date Range for Analysis")
start_date = st.date_input("Start date", df_stock.index.min().date())
end_date = st.date_input("End date", df_stock.index.max().date())
filtered = df_stock.loc[str(start_date):str(end_date)]

st.write("### Close Price (Filtered)")
st.line_chart(filtered['close'])
st.write("### Volume (Filtered)")
st.bar_chart(filtered['volume'])