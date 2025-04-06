import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
import ast
from tensorflow.keras.models import load_model
import streamlit as st
import joblib

dataset = pd.read_csv('./Carbon Emission.csv')
model = load_model('./carbon_emission_ann_final2.h5', compile=False)

def ordinal_encoder(categories, handle_unknown='use_encoded_value', unknown_value=np.nan):
    return OrdinalEncoder(categories=categories, handle_unknown=handle_unknown, unknown_value=unknown_value)

def one_hot_encoder(categories):
    return OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)

def multi_label_binarizer():
    return MultiLabelBinarizer()

def transform_column(column_name, encoder):
    return encoder.fit_transform(dataset[column_name].values.reshape(-1,  1))

def handle_multi_label_data(column_name, encoder):
    data = [ast.literal_eval(item) for item in list(dataset.pop(column_name))]
    return pd.DataFrame(encoder.fit_transform(data), columns=encoder.classes_)

encoders = {
    "Body Type": ordinal_encoder([["underweight", "normal", "overweight", "obese"]]),
    "Sex": one_hot_encoder([['male', 'female']]),
    "How Often Shower": ordinal_encoder([['less frequently', 'more frequently', 'daily', 'twice a day']]),
    "Heating Energy Source": ordinal_encoder([['electricity', 'natural gas', 'coal', 'wood']]),
    "Transport": ordinal_encoder([['walk/bicycle', 'public', 'private']]),
    "Vehicle Type": ordinal_encoder([['electric', 'hybrid', 'lpg' ,'petrol','diesel']], handle_unknown='use_encoded_value', unknown_value=-1),
    "Social Activity": ordinal_encoder([['never', 'sometimes', 'often']]),
    "Frequency of Traveling by Air": ordinal_encoder([['never', 'rarely', 'frequently', 'very frequently']]),
    "Waste Bag Size": ordinal_encoder([['small', 'medium', 'large', 'extra large']]),
    "Energy efficiency": ordinal_encoder([['No','Sometimes','Yes']]),
    "Recycling": multi_label_binarizer(),
    "Cooking_With": multi_label_binarizer(),
    "Diet": one_hot_encoder([['pescatarian', 'vegetarian', 'omnivore', 'vegan']])
}

for column, encoder in encoders.items():
    if column in ['Recycling', 'Cooking_With']:
        dataset = dataset.join(handle_multi_label_data(column, encoder))
    elif column == "Diet":
        processed = pd.DataFrame(
            transform_column(column, encoder),
            index = dataset.index,
            columns = ['pescatarian', 'vegetarian', 'omnivore', 'vegan']
        )
        dataset.pop("Diet")
        dataset = dataset.join(processed)
    else:
        dataset[column] = transform_column(column, encoder)

scaler = StandardScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

train, test = train_test_split(normalized_df, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train.drop(columns=["CarbonEmission"]),test.drop(columns=["CarbonEmission"]), train["CarbonEmission"], test["CarbonEmission"]


linearregression = LinearRegression()
decisiontreeregression = DecisionTreeRegressor(
    max_depth=10, 
    max_features=None, 
    min_samples_leaf=4, 
    min_samples_split=10, 
    random_state=42
)
supportvectorregression = SVR(C=10, epsilon=0.1, gamma=0.01, kernel='rbf')
randomforestregression = RandomForestRegressor()
xgbregression = XGBRegressor(
    learning_rate=0.1, 
    max_depth=3, 
    n_estimators=500, 
    subsample=0.8
)

linearregression.fit(X_train, y_train)
decisiontreeregression.fit(X_train, y_train)
supportvectorregression.fit(X_train, y_train)
randomforestregression.fit(X_train, y_train)
xgbregression.fit(X_train, y_train)

joblib.dump(linearregression, 'linear_regression.pkl')
joblib.dump(decisiontreeregression, 'decision_tree.pkl')
joblib.dump(supportvectorregression, 'svr.pkl')
joblib.dump(randomforestregression, 'random_forest.pkl')
joblib.dump(xgbregression, 'xgb_regressor.pkl')

y_lin = linearregression.predict(X_test)
y_dectree = decisiontreeregression.predict(X_test)
y_supvec = supportvectorregression.predict(X_test)
y_randfor = randomforestregression.predict(X_test)
y_xgb = xgbregression.predict(X_test)
y_pred = model.predict(X_test)

dataset_mean = dataset.mean()
dataset_std = dataset.std()
carbon_mean = dataset_mean.loc["CarbonEmission"]
carbon_std = dataset_std.loc["CarbonEmission"]


y_lin = (y_lin * carbon_std) + carbon_mean
y_dectree = (y_dectree * carbon_std) + carbon_mean
y_supvec = (y_supvec * carbon_std) + carbon_mean
y_randfor = (y_randfor * carbon_std) + carbon_mean
y_xgb = (y_xgb * carbon_std) + carbon_mean
y_pred = (y_pred * carbon_std) + carbon_mean

# Inverse-transform y_test and y_pred to their original scale
y_test = (y_test * carbon_std) + carbon_mean

data_actual = {
    "Regression Algorithms": ["Linear Regression", "Decision Tree Regression", 
                              "Support Vector Regression", "Random Forest Regression",
                              "XGB Regression", "ANN"],
    "Actual Carbon Footprint": [y_lin[1], y_dectree[1], y_supvec[1], 
                                y_randfor[1], y_xgb[1], y_pred[1].flatten()],
    "Actual Y": [y_test.iloc[1]]*6
}

score_actual = pd.DataFrame(data_actual)

print("Actual Carbon Footprint Predictions")
print(score_actual)

joblib.dump(encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(carbon_mean, "carbon_mean.pkl")
joblib.dump(carbon_std, "carbon_std.pkl")

# Evaluate models using R² score and MAE
print("Linear Regression R²:", r2_score(y_test, y_lin))
print("Linear Regression MAE:", mean_absolute_error(y_test, y_lin))

print("Decision Tree Regression R²:", r2_score(y_test, y_dectree))
print("Decision Tree Regression MAE:", mean_absolute_error(y_test, y_dectree))

print("Support Vector Regression R²:", r2_score(y_test, y_supvec))
print("Support Vector Regression MAE:", mean_absolute_error(y_test, y_supvec))

print("Random Forest Regression R²:", r2_score(y_test, y_randfor))
print("Random Forest Regression MAE:", mean_absolute_error(y_test, y_randfor))

print("XGB Regression R²:", r2_score(y_test, y_xgb))
print("XGB Regression MAE:", mean_absolute_error(y_test, y_xgb))

print("ANN R²:", r2_score(y_test, y_pred))
print("ANN MAE:", mean_absolute_error(y_test, y_pred))