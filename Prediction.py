import pandas as pd
import numpy as np
import csv
import json
import datetime
from pandas import DataFrame
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

use_cols=['Date','PrimaryType','District','LocationDescription']
data = pd.read_csv('/Users/srikaavya/Downloads/CrimeData.csv', usecols=use_cols, iterator=True, chunksize=100000)
data = pd.concat(data, ignore_index=True)
data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y %I:%M:%S %p"))
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
rm_dist = {1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,24,25}
data = data[data.District.isin(rm_dist)]
districts = data['District'].unique()
crimes = data['PrimaryType'].unique()
data.Date = data['Date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"))
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data.index = pd.DatetimeIndex(data.Date)
predicted_data = pd.DataFrame()
df_final = pd.DataFrame()
for district in districts:
        district_data1 = data[data.District == district]
        df_ALLCRIMES = district_data1.groupby('Date')['Date'].size().reset_index(name='ALL CRIMES')
        df_ALLCRIMES['District'] = district
        df_district = pd.DataFrame(df_ALLCRIMES, columns=['Date', 'District', 'ALL CRIMES'])
        for crime in crimes:
            df_crime = district_data1[district_data1.PrimaryType == crime]
            df_crime = df_crime.groupby(['Date'])['Date'].size().reset_index(name=crime)
            df_crime = pd.DataFrame(df_crime, columns=['Date', crime])
            df_district = pd.merge(df_district, df_crime, how='outer', on=['Date'])
        df_final = df_final.append(df_district)
        #print(df_district.info())
df_final['Date'] = pd.to_datetime(df_final.Date)
df_final.sort_values(by=['Date'], inplace=True, ascending=True)
df_final.to_csv(r'/Users/srikaavya/PycharmProjects/Diva/Project_Final/static/data/GroupedCount.csv', index=False,header=True)
data.index = pd.DatetimeIndex(data.Date)
for district in districts:
        district_data = data[data.District == district]
        data_prophet = district_data.resample('D').size().reset_index(name='Count')
        data_prophet.columns = ['Date', 'Count']
        data_prophet = data_prophet.rename(columns = {'Date':'ds','Count':'y'})
        m = Prophet()
        m.fit(data_prophet)
        future =m.make_future_dataframe(periods = 20)
        forecast = m.predict(future)
        df = forecast[ forecast['ds'] > '2020-04-09']
        df = df[['ds','yhat']].rename(columns={'ds':'Date', 'yhat':'Predicted Values'})
        df['District'] = district
        predicted_data = predicted_data.append(df)
predicted_data.to_csv(r'/Users/srikaavya/PycharmProjects/Diva/Project_Final/static/data/Prediction.csv', index = False, header=True)
data = pd.read_csv('/Users/srikaavya/PycharmProjects/Diva/Project_Final/static/data/GroupedCount.csv')
data_predict = pd.read_csv('/Users/srikaavya/PycharmProjects/Diva/Project_Final/static/data/Prediction.csv')
data = pd.DataFrame(data=data)
data_predict = pd.DataFrame(data=data_predict)
merged_df = pd.merge(data,data_predict, how='outer', on=['Date', 'District'])
print(merged_df.head())
merged_df.to_csv(r'/Users/srikaavya/PycharmProjects/Diva/Project_Final/static/data/Merged_Prediction.csv', index=False,header=True)
