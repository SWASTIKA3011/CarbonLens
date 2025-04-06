import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

# Load dataset
DATA_PATH = "/Users/Swastika/Downloads/Voluntary-Registry-Offsets-Database--v2024-12-year-end.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="PROJECTS", skiprows=3)


# Data Preprocessing
df['Project Age'] = 2025 - df['First Year of Project (Vintage)']
df['Utilization Rate'] = df['Total Credits Retired'] / df['Total Credits Issued']
df['Utilization Rate'].fillna(0, inplace=True)
df['Cost Efficiency'] = df['Estimated Annual Emission Reductions'] / (df['Total Credits Issued'] + 1)

features = ['Project Age', 'Estimated Annual Emission Reductions', 'Utilization Rate', 'Cost Efficiency']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

df['Ranking Score'] = (df['Total Credits Retired'] * 0.7 + df['Estimated Annual Emission Reductions'] * 0.3)
df['Ranking Score'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['Ranking Score'].fillna(0, inplace=True)
df['Ranking Score'] = scaler.fit_transform(df[['Ranking Score']])

# Train Model
X = df[features]
y = df['Ranking Score']
y = (y * 100).round()
y = np.clip(y, 1, 31).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRanker(objective='rank:ndcg', booster='gbtree', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train, group=[len(X_train)], eval_set=[(X_test, y_test)], eval_group=[[len(X_test)]], verbose=False)

# Streamlit UI
st.title("🌍 Carbon Offset Project Recommender")
st.subheader("Find the best carbon offset projects based on your preferences.")

# User Input for Custom Filters
region = st.selectbox("Select Region", options=["All"] + list(df['Region'].dropna().unique()))
project_type = st.selectbox("Select Project Type", options=["All"] + list(df['Type'].dropna().unique()))
top_n = st.slider("Number of recommendations", min_value=5, max_value=50, value=10)

if st.button("Get Recommendations"):
    filtered_df = df.copy()
    if region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == region]
    if project_type != "All":
        filtered_df = filtered_df[filtered_df['Type'] == project_type]
    
    filtered_df['Predicted Rank'] = model.predict(filtered_df[features])
    recommendations = filtered_df.sort_values(by='Predicted Rank', ascending=False).head(top_n)
    st.subheader("Top Carbon Offset Projects")
    st.dataframe(recommendations[['Project Name', 'Region', 'Country', 'Estimated Annual Emission Reductions', 'Predicted Rank']])



def credit():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import ndcg_score

    # Load dataset
    DATA_PATH = "/Users/Swastika/Downloads/Voluntary-Registry-Offsets-Database--v2024-12-year-end.xlsx"
    df = pd.read_excel(DATA_PATH, sheet_name="PROJECTS", skiprows=3)


    # Data Preprocessing
    df['Project Age'] = 2025 - df['First Year of Project (Vintage)']
    df['Utilization Rate'] = df['Total Credits Retired'] / df['Total Credits Issued']
    df['Utilization Rate'].fillna(0, inplace=True)
    df['Cost Efficiency'] = df['Estimated Annual Emission Reductions'] / (df['Total Credits Issued'] + 1)

    features = ['Project Age', 'Estimated Annual Emission Reductions', 'Utilization Rate', 'Cost Efficiency']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    df['Ranking Score'] = (df['Total Credits Retired'] * 0.7 + df['Estimated Annual Emission Reductions'] * 0.3)
    df['Ranking Score'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Ranking Score'].fillna(0, inplace=True)
    df['Ranking Score'] = scaler.fit_transform(df[['Ranking Score']])

    # Train Model
    X = df[features]
    y = df['Ranking Score']
    y = (y * 100).round()
    y = np.clip(y, 1, 31).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRanker(objective='rank:ndcg', booster='gbtree', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train, group=[len(X_train)], eval_set=[(X_test, y_test)], eval_group=[[len(X_test)]], verbose=False)

    # Streamlit UI
    st.title("🌍 Carbon Offset Project Recommender")
    st.subheader("Find the best carbon offset projects based on your preferences.")

    # User Input for Custom Filters
    region = st.selectbox("Select Region", options=["All"] + list(df['Region'].dropna().unique()))
    project_type = st.selectbox("Select Project Type", options=["All"] + list(df['Type'].dropna().unique()))
    top_n = st.slider("Number of recommendations", min_value=5, max_value=50, value=10)

    if st.button("Get Recommendations"):
        filtered_df = df.copy()
        if region != "All":
            filtered_df = filtered_df[filtered_df['Region'] == region]
        if project_type != "All":
            filtered_df = filtered_df[filtered_df['Type'] == project_type]
        
        filtered_df['Predicted Rank'] = model.predict(filtered_df[features])
        recommendations = filtered_df.sort_values(by='Predicted Rank', ascending=False).head(top_n)
        st.subheader("Top Carbon Offset Projects")
        st.dataframe(recommendations[['Project Name', 'Region', 'Country', 'Estimated Annual Emission Reductions', 'Predicted Rank']])
