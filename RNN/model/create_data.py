from helper import *
from pandas import HDFStore
import dill
import pandas as pd
import numpy as np
from tqdm import tqdm


columns = [
    'CompetitionDistance',
    'Promo2',
    'Open',
    'Promo',
    'StateHoliday_a',
    'StateHoliday_b',
    'StateHoliday_c',
    'StateHoliday_0',
    'Assortment_a',
    'Assortment_b',
    'Assortment_c',
    'Assortment_nan',
    'StoreType_a',
    'StoreType_b',
    'StoreType_c',
    'StoreType_d',
    'StoreType_nan',
    'DayOfWeek_1.0',
    'DayOfWeek_2.0',
    'DayOfWeek_3.0',
    'DayOfWeek_4.0',
    'DayOfWeek_5.0',
    'DayOfWeek_6.0',
    'DayOfWeek_7.0',
    'WeekOfMonth_1.0',
    'WeekOfMonth_2.0',
    'WeekOfMonth_3.0',
    'WeekOfMonth_4.0',
    'WeekOfMonth_5.0',
    'WeekOfMonth_6.0',
    'Month_1.0',
    'Month_2.0',
    'Month_3.0',
    'Month_4.0',
    'Month_5.0',
    'Month_6.0',
    'Month_7.0',
    'Month_8.0',
    'Month_9.0',
    'Month_10.0',
    'Month_11.0',
    'Month_12.0',
    'SchoolHoliday',
    'Year_1.0',
    'Year_2.0',
    'Year_3.0',
    'MeanSales',
    'MeanCustomers',
    'MeanDayOfWeekSales1',
    'MeanDayOfWeekSales2',
    'MeanDayOfWeekSales3',
    'MeanDayOfWeekSales4',
    'MeanDayOfWeekSales5',
    'MeanDayOfWeekSales6',
    'MeanDayOfWeekSales7',
    'MeanMonthSales1',
    'MeanMonthSales10',
    'MeanMonthSales11',
    'MeanMonthSales12',
    'MeanMonthSales2',
    'MeanMonthSales3',
    'MeanMonthSales4',
    'MeanMonthSales5',
    'MeanMonthSales6',
    'MeanMonthSales7',
    'MeanMonthSales8',
    'MeanMonthSales9',
    'MeanSalesNotPromo',
    'MeanSalesPromo',
    'MeanHolidaySales0',
    'MeanHolidaySales1',
    'MeanHolidaySales2',
    'MeanHolidaySales3']

print('Loading data ...')
data_dir = '../../data/'
hdf = HDFStore(data_dir + 'data.h5')
data_train = hdf['data_train']
data_train['Date'] = pd.to_datetime(data_train.Date)
data_train = data_train.ix[pd.to_datetime(data_train.Date).sort_values().index]
(DataTrain, DataTest) = split_test_train(data_train, 0.00)

print('Getting data ...')
stores = DataTrain['Store'].unique()

big_x = []
big_y = []
i = 0
print('Creating data...')
for store in tqdm(stores):
    i = i + 1
    print(i)
    data = DataTrain[DataTrain.Store == store]
    x, y = obtain_data_sequence(data, columns, n_prev=7)
    big_x.append(x)
    big_y.append(y)

dill.dump(np.array(big_x), open('bigx.dill', 'wb'))
dill.dump(np.array(big_y), open('bigy.dill', 'wb'))

print('Done ...')
