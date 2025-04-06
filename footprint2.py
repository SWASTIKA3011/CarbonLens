# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import ast
# from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
# from tensorflow.keras.models import load_model
# from sklearn.model_selection import train_test_split

# # Load models
# linearregression = joblib.load('linear_regression.pkl')
# decisiontreeregression = joblib.load('decision_tree.pkl')
# supportvectorregression = joblib.load('svr.pkl')
# randomforestregression = joblib.load('random_forest.pkl')
# xgbregression = joblib.load('xgb_regressor.pkl')
# ann_model = load_model('./carbon_emission_ann_final2.h5', compile=False)

# # Load encoders and scaler
# encoders = joblib.load("encoders.pkl")
# scaler = joblib.load("scaler.pkl")

# # Load dataset mean and std for inverse transformation
# dataset = pd.read_csv('./Carbon Emission.csv')  # Used for getting means & stds
# carbon_mean = joblib.load('carbon_mean.pkl')
# carbon_std = joblib.load('carbon_std.pkl')
# X_train = joblib.load('X_train.pkl')

# # Streamlit UI
# st.title("Carbon Emission Prediction App")
# st.sidebar.header("Select Model")
# model_choice = st.sidebar.selectbox("Choose a model:", [
#     "Linear Regression", "Decision Tree", "Support Vector Regression", "Random Forest", "XGBoost", "Artificial Neural Network"])

# # User input fields
# def user_input():
#     return {
#         "Body Type": st.selectbox("Body Type", ["underweight", "normal", "overweight", "obese"]),
#         "Sex": st.selectbox("Sex", ["male", "female"]),
#         "Diet": st.selectbox("Diet", ["pescatarian", "vegetarian", "omnivore", "vegan"]),
#         "How Often Shower": st.selectbox("Shower Frequency", ["less frequently", "more frequently", "daily", "twice a day"]),
#         "Heating Energy Source": st.selectbox("Heating Source", ["electricity", "natural gas", "coal", "wood"]),
#         "Transport": st.selectbox("Transport Mode", ["walk/bicycle", "public", "private"]),
#         "Vehicle Type": st.selectbox("Vehicle Type", ["electric", "hybrid", "lpg", "petrol", "diesel"]),
#         "Social Activity": st.selectbox("Social Activity", ["never", "sometimes", "often"]),
#         "Monthly Grocery Bill": st.number_input("Grocery Bill (USD)", min_value=0, value=1000),
#         "Frequency of Traveling by Air": st.selectbox("Air Travel Frequency", ["never", "rarely", "frequently", "very frequently"]),
#         "Vehicle Monthly Distance Km": st.number_input("Monthly Distance (km)", min_value=0, value=1500),
#         "Waste Bag Size": st.selectbox("Waste Bag Size", ["small", "medium", "large", "extra large"]),
#         "Waste Bag Weekly Count": st.number_input("Waste Bags per Week", min_value=0, value=6),
#         "How Long TV PC Daily Hour": st.number_input("TV/PC Hours Daily", min_value=0, value=16),
#         "How Many New Clothes Monthly": st.number_input("New Clothes per Month", min_value=0, value=10),
#         "How Long Internet Daily Hour": st.number_input("Internet Hours Daily", min_value=0, value=18),
#         "Energy efficiency": st.selectbox("Energy Efficiency", ["No", "Sometimes", "Yes"]),
#         "Recycling": st.multiselect("Recycling", ["Paper", "Plastic", "Glass", "Metal"]),
#         "Cooking_With": st.multiselect("Cooking Appliances", ["Stove", "Microwave", "Oven", "Grill", "Airfryer"])
#     }

# data = user_input()

# def preprocess_new_data(new_data):
#     """
#     Transforms new data using the same preprocessing steps applied to training data.
#     """
#     new_df = pd.DataFrame([new_data])

#     required_columns = X_train.columns
#     missing_columns = set(required_columns) - set(new_df.columns)

#     for column, encoder in encoders.items():
#         print(f"Processing column: {column}")  # Debugging print
        
#         if column in ['Recycling', 'Cooking_With']:
#             if isinstance(new_df[column].iloc[0], str):
#                 print(f"{column} column before transformation:", new_df[column])  # Debugging print
#                 new_df[column] = new_df[column].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
#             # Handle multi-label transformation
#             encoded_df = pd.DataFrame(encoder.transform(new_df[column]), columns=encoder.classes_)
#             new_df = new_df.join(encoded_df).drop(columns=[column])
#             print(f"Columns after encoding {column}:", new_df.columns)  # Debugging print
            
#         elif column == "Diet":
#             print(f"Diet column before transformation:", new_df[column])  # Debugging print
#             encoded = pd.DataFrame(
#                 encoder.transform(new_df[[column]]),
#                 index=new_df.index,
#                 columns=['pescatarian', 'vegetarian', 'omnivore', 'vegan']
#             )
#             new_df = new_df.drop(columns=[column]).join(encoded)
#             print("Columns after encoding Diet:", new_df.columns)  # Debugging print
#         else:
#             new_df[column] = encoder.transform(new_df[[column]])

#     # Add missing columns with zeros to prevent errors during column alignment
#     for col in missing_columns:
#         new_df[col] = 0
    
#     new_df = new_df[required_columns]
    
#     print("New dataframe columns after reordering:", new_df.columns) 

#     # Normalize new data based on training set scaler
#     new_df = pd.DataFrame(scaler.transform(new_df), columns=X_train.columns)

#     return new_df



# def predict_carbon_footprint(new_data):
#     """
#     Predicts carbon footprint for new data using all trained models.
#     """

#     # Generate predictions
#     y_lin = linearregression.predict(new_data)
#     y_dectree = decisiontreeregression.predict(new_data)
#     y_supvec = supportvectorregression.predict(new_data)
#     y_randfor = randomforestregression.predict(new_data)
#     y_xgb = xgbregression.predict(new_data)
#     y_ann = ann_model.predict(new_data)

#     y_ann = (y_ann * carbon_std) + carbon_mean

#     y_ann = y_ann[0][0]

#     return {
#         "Linear Regression": y_lin[0],
#         "Decision Tree": y_dectree[0],
#         "Support Vector Regression": y_supvec[0],
#         "Random Forest": y_randfor[0],
#         "XGBoost": y_xgb[0],
#         "Artificial Neural Network": y_ann
#     }


# if st.sidebar.button("Predict"):
#     new_data = preprocess_new_data(data)
#     carbon_predictions = predict_carbon_footprint(new_data)

#     print("\n🔍 **Carbon Footprint Predictions for New Data:**")
#     for model, value in carbon_predictions.items():
#         if model in model_choice:
#             st.write(f"Predicted Carbon Emission for {model}: {value:.2f} kg CO₂")



#######################################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

linearregression = joblib.load('linear_regression.pkl')
decisiontreeregression = joblib.load('decision_tree.pkl')
supportvectorregression = joblib.load('svr.pkl')
randomforestregression = joblib.load('random_forest.pkl')
xgbregression = joblib.load('xgb_regressor.pkl')
ann_model = load_model('./carbon_emission_ann_final2.h5', compile=False)

encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

dataset = pd.read_csv('./Carbon Emission.csv')  
carbon_mean = joblib.load('carbon_mean.pkl')
carbon_std = joblib.load('carbon_std.pkl')
X_train = joblib.load('X_train.pkl')

st.title("Carbon Emission Prediction App")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["👴 Personal", "🚗 Travel", "🗑️ Waste", "⚡ Energy", "💸 Consumption"])

with tab1:
    height = st.number_input("Height (in cm)", 0, 251, value=None, placeholder="160", help="in cm")
    weight = st.number_input("Weight (in kg)", 0, 250, value=None, placeholder="75", help="in kg")
    
    if weight is None or weight == 0:
        weight = 1
    if height is None or height == 0:
        height = 1
        
    calculation = weight / (height / 100) ** 2
    body_type = "underweight" if calculation < 18.5 else "normal" if 18.5 <= calculation < 25 else "overweight" if 25 <= calculation < 30 else "obese"
    
    sex = st.selectbox('Gender', ["female", "male"])
    diet = st.selectbox('Diet', ['omnivore', 'pescatarian', 'vegetarian', 'vegan'], help="Omnivore: Eats both plants and animals.")
    social = st.selectbox('Social Activity', ['never', 'often', 'sometimes'], help="How often do you go out?")

with tab2:
    transport = st.selectbox('Transportation', ['public', 'private', 'walk/bicycle'], help="Which transportation method do you prefer?")
    if transport == "private":
        vehicle_type = st.selectbox('Vehicle Type', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric'], help="What type of fuel do you use?")
    else:
        vehicle_type = "None"

    if transport == "walk/bicycle":
        vehicle_km = 0
    else:
        vehicle_km = st.slider('Monthly distance traveled by vehicle (km)', 0, 5000, 0)

    air_travel = st.selectbox('Air Travel Frequency', ['never', 'rarely', 'frequently', 'very frequently'], help="How often did you fly last month?")

with tab3:
    waste_bag = st.selectbox('Waste Bag Size', ['small', 'medium', 'large', 'extra large'])
    waste_count = st.slider('Waste bags disposed weekly', 0, 10, 0)
    recycle = st.multiselect('Do you recycle?', ['Plastic', 'Paper', 'Metal', 'Glass'])

with tab4:
    heating_energy = st.selectbox('Heating Energy Source', ['natural gas', 'electricity', 'wood', 'coal'])
    for_cooking = st.multiselect('Cooking Systems Used', ['microwave', 'oven', 'grill', 'airfryer', 'stove'])
    energy_efficiency = st.selectbox('Energy Efficiency Consideration', ['No', 'Yes', 'Sometimes'])
    daily_tv_pc = st.slider('Hours spent in front of TV/PC daily', 0, 24, 0)
    internet_daily = st.slider('Daily internet usage (hours)', 0, 24, 0)

with tab5:
    shower = st.selectbox('Shower Frequency', ['daily', 'twice a day', 'more frequently', 'less frequently'])
    grocery_bill = st.slider('Monthly Grocery Bill ($)', 0, 500, 0)
    clothes_monthly = st.slider('Clothes purchased monthly', 0, 30, 0)


def user_input():
    return {
        "Body Type": body_type,
        "Sex": sex,
        "Diet": diet,
        "How Often Shower": shower,
        "Heating Energy Source": heating_energy,
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Social Activity": social,
        "Monthly Grocery Bill": grocery_bill,
        "Frequency of Traveling by Air": air_travel,
        "Vehicle Monthly Distance Km": vehicle_km,
        "Waste Bag Size": waste_bag,
        "Waste Bag Weekly Count": waste_count,
        "How Long TV PC Daily Hour": daily_tv_pc,
        "How Many New Clothes Monthly": clothes_monthly,
        "How Long Internet Daily Hour": internet_daily,
        "Energy efficiency": energy_efficiency,
        "Recycling": recycle,
        "Cooking_With": for_cooking
    }

data = user_input()

def preprocess_new_data(new_data):
    """
    Transforms new data using the same preprocessing steps applied to training data.
    """
    new_df = pd.DataFrame([new_data])

    required_columns = X_train.columns
    missing_columns = set(required_columns) - set(new_df.columns)

    for column, encoder in encoders.items():
        print(f"Processing column: {column}")  
        
        if column in ['Recycling', 'Cooking_With']:
            if isinstance(new_df[column].iloc[0], str):
                print(f"{column} column before transformation:", new_df[column])  
                new_df[column] = new_df[column].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

            encoded_df = pd.DataFrame(encoder.transform(new_df[column]), columns=encoder.classes_)
            new_df = new_df.join(encoded_df).drop(columns=[column])
            print(f"Columns after encoding {column}:", new_df.columns)  
            
        elif column == "Diet":
            print(f"Diet column before transformation:", new_df[column])  
            encoded = pd.DataFrame(
                encoder.transform(new_df[[column]]),
                index=new_df.index,
                columns=['pescatarian', 'vegetarian', 'omnivore', 'vegan']
            )
            new_df = new_df.drop(columns=[column]).join(encoded)
            print("Columns after encoding Diet:", new_df.columns)  
        else:
            new_df[column] = encoder.transform(new_df[[column]])

    for col in missing_columns:
        new_df[col] = 0
    
    new_df = new_df[required_columns]
    
    print("New dataframe columns after reordering:", new_df.columns) 

    new_df = pd.DataFrame(scaler.transform(new_df), columns=X_train.columns)

    return new_df


def predict_carbon_footprint(new_data):
    """
    Predicts carbon footprint for new data using all trained models.
    """

    y_lin = linearregression.predict(new_data)
    y_dectree = decisiontreeregression.predict(new_data)
    y_supvec = supportvectorregression.predict(new_data)
    y_randfor = randomforestregression.predict(new_data)
    y_xgb = xgbregression.predict(new_data)
    y_ann = ann_model.predict(new_data)

    y_ann = (y_ann * carbon_std) + carbon_mean

    y_ann = y_ann[0][0]

    return {
        "Linear Regression": y_lin[0],
        "Decision Tree": y_dectree[0],
        "Support Vector Regression": y_supvec[0],
        "Random Forest": y_randfor[0],
        "XGBoost": y_xgb[0],
        "Artificial Neural Network": y_ann
    }
st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)

model_choice = st.multiselect("Choose a mode/models:", [
"Linear Regression", "Decision Tree", "Support Vector Regression", "Random Forest", "XGBoost", "Artificial Neural Network"])


with st.expander("Model Performance Metrics"):  
    metrics = {
        "Model": ["Linear Regression", "Decision Tree Regression", "Support Vector Regression", "Random Forest Regression", "XGB Regression", "ANN"],
        "R²": [0.8504363662643909, 0.8555275662089159, 0.6868458326328166, 0.926513427349288, 0.9889962673187256, 0.987363874912262],
        "MAE": [279.3849742823618, 292.7702537058985, 373.50352118719087, 208.65418000000003, 78.4939956665039, 84.876708984375]
    }

    df_metrics = pd.DataFrame(metrics)

    st.write("#### Model Performance Summary")
    st.table(df_metrics)
    st.write(" ")

    best_model = df_metrics.loc[df_metrics['R²'].idxmax()]
    st.write(f"#### Best Performing Model: **{best_model['Model']}**")
    st.write(f"R²: **{best_model['R²']:.4f}**, MAE: **{best_model['MAE']:.2f}**")
    st.write(" ")

    st.write("#### Model Performance Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].bar(df_metrics['Model'], df_metrics['R²'])
    ax[0].set_title('R² Values')
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].bar(df_metrics['Model'], df_metrics['MAE'])
    ax[1].set_title('MAE Values')
    ax[1].tick_params(axis='x', rotation=90)

    st.pyplot(fig)

if st.button("Predict"):
    new_data = preprocess_new_data(data)
    carbon_predictions = predict_carbon_footprint(new_data)

    print("\n🔍 **Carbon Footprint Predictions for New Data:**")
    for model, value in carbon_predictions.items():
        if model in model_choice:
            st.write(f"Predicted Carbon Emission for **{model}**: **{value:.2f}** kg CO₂")

############################################

def footprint():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import ast
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
    from tensorflow.keras.models import load_model
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt

    linearregression = joblib.load('linear_regression.pkl')
    decisiontreeregression = joblib.load('decision_tree.pkl')
    supportvectorregression = joblib.load('svr.pkl')
    randomforestregression = joblib.load('random_forest.pkl')
    xgbregression = joblib.load('xgb_regressor.pkl')
    ann_model = load_model('./carbon_emission_ann_final2.h5', compile=False)

    encoders = joblib.load("encoders.pkl")
    scaler = joblib.load("scaler.pkl")

    dataset = pd.read_csv('./Carbon Emission.csv')  
    carbon_mean = joblib.load('carbon_mean.pkl')
    carbon_std = joblib.load('carbon_std.pkl')
    X_train = joblib.load('X_train.pkl')

    st.title("Carbon Emission Prediction App")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👴 Personal", "🚗 Travel", "🗑️ Waste", "⚡ Energy", "💸 Consumption"])

    with tab1:
        height = st.number_input("Height (in cm)", 0, 251, value=None, placeholder="160", help="in cm")
        weight = st.number_input("Weight (in kg)", 0, 250, value=None, placeholder="75", help="in kg")
        
        if weight is None or weight == 0:
            weight = 1
        if height is None or height == 0:
            height = 1
            
        calculation = weight / (height / 100) ** 2
        body_type = "underweight" if calculation < 18.5 else "normal" if 18.5 <= calculation < 25 else "overweight" if 25 <= calculation < 30 else "obese"
        
        sex = st.selectbox('Gender', ["female", "male"])
        diet = st.selectbox('Diet', ['omnivore', 'pescatarian', 'vegetarian', 'vegan'], help="Omnivore: Eats both plants and animals.")
        social = st.selectbox('Social Activity', ['never', 'often', 'sometimes'], help="How often do you go out?")

    with tab2:
        transport = st.selectbox('Transportation', ['public', 'private', 'walk/bicycle'], help="Which transportation method do you prefer?")
        if transport == "private":
            vehicle_type = st.selectbox('Vehicle Type', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric'], help="What type of fuel do you use?")
        else:
            vehicle_type = "None"

        if transport == "walk/bicycle":
            vehicle_km = 0
        else:
            vehicle_km = st.slider('Monthly distance traveled by vehicle (km)', 0, 5000, 0)

        air_travel = st.selectbox('Air Travel Frequency', ['never', 'rarely', 'frequently', 'very frequently'], help="How often did you fly last month?")

    with tab3:
        waste_bag = st.selectbox('Waste Bag Size', ['small', 'medium', 'large', 'extra large'])
        waste_count = st.slider('Waste bags disposed weekly', 0, 10, 0)
        recycle = st.multiselect('Do you recycle?', ['Plastic', 'Paper', 'Metal', 'Glass'])

    with tab4:
        heating_energy = st.selectbox('Heating Energy Source', ['natural gas', 'electricity', 'wood', 'coal'])
        for_cooking = st.multiselect('Cooking Systems Used', ['microwave', 'oven', 'grill', 'airfryer', 'stove'])
        energy_efficiency = st.selectbox('Energy Efficiency Consideration', ['No', 'Yes', 'Sometimes'])
        daily_tv_pc = st.slider('Hours spent in front of TV/PC daily', 0, 24, 0)
        internet_daily = st.slider('Daily internet usage (hours)', 0, 24, 0)

    with tab5:
        shower = st.selectbox('Shower Frequency', ['daily', 'twice a day', 'more frequently', 'less frequently'])
        grocery_bill = st.slider('Monthly Grocery Bill ($)', 0, 500, 0)
        clothes_monthly = st.slider('Clothes purchased monthly', 0, 30, 0)


    def user_input():
        return {
            "Body Type": body_type,
            "Sex": sex,
            "Diet": diet,
            "How Often Shower": shower,
            "Heating Energy Source": heating_energy,
            "Transport": transport,
            "Vehicle Type": vehicle_type,
            "Social Activity": social,
            "Monthly Grocery Bill": grocery_bill,
            "Frequency of Traveling by Air": air_travel,
            "Vehicle Monthly Distance Km": vehicle_km,
            "Waste Bag Size": waste_bag,
            "Waste Bag Weekly Count": waste_count,
            "How Long TV PC Daily Hour": daily_tv_pc,
            "How Many New Clothes Monthly": clothes_monthly,
            "How Long Internet Daily Hour": internet_daily,
            "Energy efficiency": energy_efficiency,
            "Recycling": recycle,
            "Cooking_With": for_cooking
        }

    data = user_input()

    def preprocess_new_data(new_data):
        """
        Transforms new data using the same preprocessing steps applied to training data.
        """
        new_df = pd.DataFrame([new_data])

        required_columns = X_train.columns
        missing_columns = set(required_columns) - set(new_df.columns)

        for column, encoder in encoders.items():
            print(f"Processing column: {column}")  
            
            if column in ['Recycling', 'Cooking_With']:
                if isinstance(new_df[column].iloc[0], str):
                    print(f"{column} column before transformation:", new_df[column])  
                    new_df[column] = new_df[column].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

                encoded_df = pd.DataFrame(encoder.transform(new_df[column]), columns=encoder.classes_)
                new_df = new_df.join(encoded_df).drop(columns=[column])
                print(f"Columns after encoding {column}:", new_df.columns)  
                
            elif column == "Diet":
                print(f"Diet column before transformation:", new_df[column])  
                encoded = pd.DataFrame(
                    encoder.transform(new_df[[column]]),
                    index=new_df.index,
                    columns=['pescatarian', 'vegetarian', 'omnivore', 'vegan']
                )
                new_df = new_df.drop(columns=[column]).join(encoded)
                print("Columns after encoding Diet:", new_df.columns)  
            else:
                new_df[column] = encoder.transform(new_df[[column]])

        for col in missing_columns:
            new_df[col] = 0
        
        new_df = new_df[required_columns]
        
        print("New dataframe columns after reordering:", new_df.columns) 

        new_df = pd.DataFrame(scaler.transform(new_df), columns=X_train.columns)

        return new_df


    def predict_carbon_footprint(new_data):
        """
        Predicts carbon footprint for new data using all trained models.
        """

        y_lin = linearregression.predict(new_data)
        y_dectree = decisiontreeregression.predict(new_data)
        y_supvec = supportvectorregression.predict(new_data)
        y_randfor = randomforestregression.predict(new_data)
        y_xgb = xgbregression.predict(new_data)
        y_ann = ann_model.predict(new_data)

        y_ann = (y_ann * carbon_std) + carbon_mean

        y_ann = y_ann[0][0]

        return {
            "Linear Regression": y_lin[0],
            "Decision Tree": y_dectree[0],
            "Support Vector Regression": y_supvec[0],
            "Random Forest": y_randfor[0],
            "XGBoost": y_xgb[0],
            "Artificial Neural Network": y_ann
        }
    st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)

    model_choice = st.multiselect("Choose a mode/models:", [
    "Linear Regression", "Decision Tree", "Support Vector Regression", "Random Forest", "XGBoost", "Artificial Neural Network"])


    with st.expander("Model Performance Metrics"):  
        metrics = {
            "Model": ["Linear Regression", "Decision Tree Regression", "Support Vector Regression", "Random Forest Regression", "XGB Regression", "ANN"],
            "R²": [0.8504363662643909, 0.8555275662089159, 0.6868458326328166, 0.926513427349288, 0.9889962673187256, 0.987363874912262],
            "MAE": [279.3849742823618, 292.7702537058985, 373.50352118719087, 208.65418000000003, 78.4939956665039, 84.876708984375]
        }

        df_metrics = pd.DataFrame(metrics)

        st.write("#### Model Performance Summary")
        st.table(df_metrics)
        st.write(" ")

        best_model = df_metrics.loc[df_metrics['R²'].idxmax()]
        st.write(f"#### Best Performing Model: **{best_model['Model']}**")
        st.write(f"R²: **{best_model['R²']:.4f}**, MAE: **{best_model['MAE']:.2f}**")
        st.write(" ")

        st.write("#### Model Performance Visualization")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].bar(df_metrics['Model'], df_metrics['R²'])
        ax[0].set_title('R² Values')
        ax[0].tick_params(axis='x', rotation=90)

        ax[1].bar(df_metrics['Model'], df_metrics['MAE'])
        ax[1].set_title('MAE Values')
        ax[1].tick_params(axis='x', rotation=90)

        st.pyplot(fig)

    if st.button("Predict"):
        new_data = preprocess_new_data(data)
        carbon_predictions = predict_carbon_footprint(new_data)

        print("\n🔍 **Carbon Footprint Predictions for New Data:**")
        for model, value in carbon_predictions.items():
            if model in model_choice:
                st.write(f"Predicted Carbon Emission for **{model}**: **{value:.2f}** kg CO₂")