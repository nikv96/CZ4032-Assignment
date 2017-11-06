# xgboost_regression_log.py
"""
Private Score: 0.12727
"""

import datetime as dt
import pickle
import sys

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None

################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################
training_df = pd.read_csv("data/train.csv", parse_dates=[2], dtype={"StateHoliday": pd.np.string_})
store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv", parse_dates=[3], dtype={"StateHoliday": pd.np.string_})

################################################################
# Process Data (Universal)                                     #
################################################################

def is_nan(val):
    return val != val


def less_than_ten(val):
    if int(val) < 10:
        return "0" + val
    else:
        return val

############################################
# training_df & test_df                    #
############################################

# Fill NaN values in test_df with Open = 1 if DayOfWeek != 7
test_df["Open"][is_nan(test_df["Open"])] = (test_df["DayOfWeek"] != 7).astype(int)
training_df["Open"][is_nan(training_df["Open"])] = (training_df["DayOfWeek"] != 7).astype(int)

# Create "Year" & "Month" columns
training_df["Year"] = training_df["Date"].dt.year
training_df["Month"] = training_df["Date"].dt.month

test_df["Year"] = test_df["Date"].dt.year
test_df["Month"] = test_df["Date"].dt.month

# "StateHoliday" has values "0" & 0
training_df["StateHoliday"].loc[training_df["StateHoliday"] == 0] = "0"
test_df["StateHoliday"].loc[test_df["StateHoliday"] == 0] = "0"

############################################
# store_df                                 #
############################################

# Fill NaN values in store_df for "CompetitionDistance" = 0 (since no record exists where "CD" = NaN & "COS[Y/M]" = !NaN)
store_df["CompetitionDistance"][is_nan(store_df["CompetitionDistance"])] = 0

# Fill NaN values in store_df for "CompetitionSince[X]" with 1900-01
store_df["CompetitionOpenSinceYear"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceYear"]))] = 1900
store_df["CompetitionOpenSinceMonth"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceMonth"]))] = 1

################################################################
# Process Data (Custom)                                        #
################################################################

# Merge store and training data frames
training_df = pd.merge(training_df, store_df, on="Store", how="left")
test_df = pd.merge(test_df, store_df, on="Store", how="left")

# Creating DayOfMonth column
training_df["DayOfMonth"] = training_df.Date.dt.day
test_df["DayOfMonth"] = test_df.Date.dt.day

# Filling all NaN values with 0
training_df = training_df.fillna(0)
test_df = test_df.fillna(0)

# Using only Open Stores for training
training_df = training_df[training_df["Open"] == 1]

# Log factorization of Sales changes the distribution and makes the performance better
training_df['Sales'] = np.log1p(training_df['Sales'])

# List of features used in training
features = ["Store", "DayOfWeek", "Year", "Month", "DayOfMonth", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]

# Label encoding of columns (eg. StoreType with "a", "b", "c" and "d" would become 1, 2, 3 and 4)
for f in training_df[features]:
    if training_df[f].dtype == "object":
        labels = LabelEncoder()
        labels.fit(list(training_df[f].values) + list(test_df[f].values))
        training_df[f] = labels.transform(list(training_df[f].values))
        test_df[f] = labels.transform(list(test_df[f].values))


################################################################
# RMSPE Function                                               #
################################################################

def rmspe(y_true, y_pred):
    """
    RMSPE =  sqrt(1/n * sum( ( (y_true - y_pred)/y_true) ** 2 ) )
    """
    # multiplying_factor = 1/y_true when y_true != 0, else multiplying_factor = 0
    multiplying_factor = np.zeros(y_true.shape, dtype=float)
    indices = y_true != 0
    multiplying_factor[indices] = 1.0 / (y_true[indices])
    diff = y_true - y_pred
    diff_percentage = diff * multiplying_factor
    diff_percentage_squared = diff_percentage ** 2
    rmspe = np.sqrt(np.mean(diff_percentage_squared))
    return rmspe


################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
A XGB regression model for all stores. This model is the same as xgboostregressor2 but it uses log factorization of the output variable "Sales". This improves the distribution and prediction results.

Features: Store, DayOfWeek, Year, Month, DayOfMonth, Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo2

Assumptions:
- DayOfMonth has an effect on sales, for example, the sales is higher on paydays.
- The store's opening/closing dates does not affect the store's performance. For example, a store that was closed yesterday will not get more sales today because of that.
"""

X_train = training_df[features]
X_test = test_df[features]
y_train = training_df["Sales"]

# Comment this block when not training
################ TRAINING ###############
print("Training...")
regressor = XGBRegressor(n_estimators=3000, nthread=-1, max_depth=12,
                         learning_rate=0.02, silent=True, subsample=0.9, colsample_bytree=0.7)
regressor.fit(np.array(X_train), y_train)
with open("models/xgboost_regression_log.pkl", "wb") as f:
    pickle.dump(regressor, f)

print("Model saved to models/xgboost_regression_log.pkl")
########### TRAINING COMPLETED ##########

# Uncomment this block when not training
# with open("models/xgboost_regression_log.pkl", "rb") as f:
#     regressor = pickle.load(f)
# print ("Loaded the model.")

print("Making Predictions...")

predictions = regressor.predict(np.array(X_test))
result = pd.DataFrame({"Id": test_df["Id"], "Sales": np.expm1(predictions)})
result.to_csv("predictions/xgboost_regression_log.csv", index=False)

print("Predictions saved to predictions/xgboost_regression_log.csv.")

# Uncomment this section to show the feature importance chart
# mapper = {'f{0}'.format(i): v for i, v in enumerate(features)}
# mapped = {mapper[k]: v for k, v in regressor.booster().get_score(importance_type='weight').items()}
# plot_importance(mapped)
# plt.savefig('plots/xgboost_regression_log_feature_importance.png', dpi=800)
# plt.close('all')