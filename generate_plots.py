'''
The script generates plots for analyzing the Rossmann Store Sales dataset.

All generated plots are stored in the plots/ directory.
'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

####################### Helper Functions #######################

def is_nan(val):
    return val != val


def less_than_ten(val):
    if int(val) < 10:
        return "0" + val
    else:
        return val

################################################################

# Read the datasets
training_df = pd.read_csv("data/train.csv", dtype={"StateHoliday": pd.np.string_}, parse_dates=[2])
store_df = pd.read_csv("data/store.csv", dtype={"StateHoliday": pd.np.string_}, parse_dates=[2])

################## Preprocessing #######################

training_df["Year"] = training_df["Date"].year
training_df["Month"] = training_df["Date"].month
training_df["DayOfMonth"] = training_df["Date"].day
training_df["YearMonth"] = str(training_df["Year"]) + "-" + str(training_df["Month"])

# StateHolidayBinary
training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

# Create DataFrame for only open days
training_df_open = training_df[training_df["Open"] != 0]
# Drop "Open" column
training_df_open.drop(["Open"], axis=1, inplace=True)

# replace null elements with 1 if day of week is not 7
training_df['Open'].fillna(value=(training_df['DayOfWeek'] != 7).astype(int))
# sanity check
training_df[training_df['Open']==0].count()

# Add "AvgSales" & "AvgCustomers" columns to store_df
avg_sales_customers = training_df_open.groupby("Store")[["Sales", "Customers"]].mean()
avg_sales_customers_df = DataFrame({"Store": avg_sales_customers.index, "AvgSales": avg_sales_customers["Sales"], "AvgCustomers": avg_sales_customers["Customers"]}, columns=["Store", "AvgSales", "AvgCustomers"])
store_df = pd.merge(avg_sales_customers_df, store_df, on="Store")

# Add "AvgSalesPerCustomer" column to store_df
store_tot_sales = training_df.groupby([training_df["Store"]])["Sales"].sum()
store_tot_custs = training_df.groupby([training_df["Store"]])["Customers"].sum()
store_salespercust = store_tot_sales / store_tot_custs
store_df = pd.merge(store_df, store_salespercust.reset_index(name="AvgSalesPerCustomer"), on="Store")

# Fill NaN values in store_df for "CompetitionDistance" = 0 (since no record exists where "CD" = NaN & "COS[Y/M]" = !NaN)
store_df["CompetitionDistance"][is_nan(store_df["CompetitionDistance"])] = 0

# combining store and train df
training_store_df = pd.merge(training_df, store_df, on="Store", how="left")

########################################################

###################### Data Analysis ##########################

# general information on the training sets
print(training_store_df.info())
print(training_store_df.describe())
print(training_store_df.groupby([training_df['Store']]).describe())

################# Analysis on Zero Sales Dataset ################
# Corner case analysis
zero_sales_df = training_df[training_df['Sales'] == 0]
print('Count of Zero Sales DF')
print(zero_sales_df.count())
print('Count of Zero Sales DF where DayOfWeek is Sunday')
print(zero_sales_df[zero_sales_df['DayOfWeek'] == 7].count())
print('Count of Zero Sales DF where SchoolHoliday is 1')
print(zero_sales_df[zero_sales_df['SchoolHoliday'] == 1].count())
print('Count of Zero Sales DF where StateHoliday is not 0')
print(zero_sales_df[zero_sales_df['StateHoliday'] != '0'].count())
print('Count of Zero Sales DF where Promo is 0')
print(zero_sales_df[zero_sales_df['Promo'] == 0].count())
print('Zero sales count - zero customers count')
print(zero_sales_df.count() - training_df[training_df['Customers'] == 0].count())
plt.figure()
plt.xlabel('Date')
plt.ylabel('Customers')
plt.plot(zero_sales_df.sort_values('Date')['Date'], zero_sales_df.sort_values('Date')['Customers'], dpi=800)
plt.tight_layout()
plt.savefig('plots/Count of Customers at Zero Sales vs Date.png', dpi=800)
plt.close('all')
print('Plotted Count of Customers at Zero Sales vs Date')
########################################################


################# Analysis on StoreType ################
# Counts of Stores by Store Type
plt.figure()
plt.bar(sorted(training_store_df['StoreType'].unique()), training_store_df.groupby([training_store_df['StoreType']]).size())
plt.title('Count of Stores by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Count')
plt.savefig('plots/Count of Stores by Store Type.png', dpi=800)
plt.close('all')

# Sales of Stores by Store Type
plt.figure()
plt.bar(sorted(training_store_df['StoreType'].unique()), training_store_df.groupby([training_store_df['StoreType']]).mean()['Sales'])
plt.title('Sales of Stores by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Sales')
plt.savefig('plots/Sales Of Stores by Store Type.png', dpi=800)
plt.close('all')
print('Plotted Sales and Count by Store Type')
########################################################

################# Analysis on Assortment ################
# Counts of Stores by Assortment
plt.figure()
plt.bar(sorted(training_store_df['Assortment'].unique()), training_store_df.groupby([training_store_df['Assortment']]).size())
plt.title('Count of Stores by Assortment')
plt.xlabel('Assortment')
plt.ylabel('Count')
plt.savefig('plots/Count of Stores by Assortment.png', dpi=800)
plt.close('all')

# Sales of Stores by Assortment
plt.figure()
plt.bar(sorted(training_store_df['Assortment'].unique()), training_store_df.groupby([training_store_df['Assortment']]).mean()['Sales'])
plt.title('Sales of Stores by Assortment')
plt.xlabel('Assortment')
plt.ylabel('Sales')
plt.savefig('plots/Sales Of Stores by Assortment.png', dpi=800)
plt.close('all')
print('Plotted Sales and Count by Assortment')
########################################################

################# Analysis on Promo2 ################
# count of stores with promo2 active/inactive
plt.figure()
plt.bar(['0', '1'], training_store_df.groupby([training_store_df['Promo2']]).size())
plt.title('Count of Stores by Promo2')
plt.xlabel('Promo2')
plt.ylabel('Count')
plt.savefig('plots/Count & Sales Of Stores by Promo2.png', dpi=800)
plt.close('all')

# count of stores with promo2 active/inactive
plt.figure()
plt.bar(['0', '1'], training_store_df.groupby([training_store_df['Promo2']]).sum()['Sales'])
plt.title('Sales of Stores by Promo2')
plt.xlabel('Promo2')
plt.ylabel('Sales')
plt.savefig('plots/Sales Of Stores by Promo2.png', dpi=800)
plt.close('all')
print('Plotted Sales and Count by Promo2')
########################################################

############## Analysis on Date Derivatives #############
# Generate plots for Avg. Sales & Percentage Change (by Year-Month)
average_sales = training_df.groupby("YearMonth")["Sales"].mean()
pct_change_sales = training_df.groupby("YearMonth")["Sales"].sum().pct_change()
fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(15, 16))
ax1 = average_sales.plot(legend=True, ax=axis1, marker="o", title="Average Sales")
ax1.set_xticks(range(len(average_sales)))
ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)
ax2 = pct_change_sales.plot(legend=True, ax=axis2, marker="o", rot=90, colormap="summer", title="Sales Percent Change")
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Percentage Change (by Year-Month).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Percentage Change (by Year-Month)")

# Generate plots for Avg. Sales & Customers (by Year)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="Year", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="Year", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Year).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Year)")

# Generate plots for Avg. Sales & Customers (by Month)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="Month", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="Month", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Month).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Month)")

# Generate plots for Avg. Sales & Customers (by Day of Month)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 8))
sns.barplot(x="DayOfMonth", y="Sales", data=training_df, ax=axis1, ci=None)
sns.barplot(x="DayOfMonth", y="Customers", data=training_df, ax=axis2, ci=None)
fig.tight_layout()
fig.savefig("plots/Avg. Sales & Customers (by Day Of Month).png", dpi=fig.dpi)
fig.clf()
plt.close(fig)
print("Plotted Avg. Sales & Customers (by Day Of Month)")

