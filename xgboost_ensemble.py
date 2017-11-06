"""
Score: 0.11880, i = 0.67, j = 0.970
"""

import pandas as pd
import numpy as np
import sys

pd.options.mode.chained_assignment = None

determineBestWeights = False

################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################

file1 = "predictions/xgboost_regression_log3.csv"
file2 = "predictions/xgboost_regression_log.csv"

model1 = pd.read_csv(file1)
model2 = pd.read_csv(file2)

################################################################
# Process Data                                                 #
################################################################

# Get the predictions from the both the datasets and the true sales value
sales_model1 = model1["Sales"]
sales_model2 = model2["Sales"]


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
# Making Predictions                                           #
################################################################
"""
This model ensembles two xgboost models using static combination with weighted averages.
"""

# Predictions using static combining
i = 0.67
j = 0.97
sales = (sales_model1 * i + sales_model2 * (1.0 - i)) * j

result = pd.DataFrame({"Id": model2["Id"], "Sales": sales})
result.to_csv("predictions/xgboost_ensemble.csv", index=False)
print("Predictions saved to predictions/xgboost_ensemble.csv.")