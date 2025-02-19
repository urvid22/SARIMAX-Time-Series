#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

url = r"C:\Users\urvi9\MVA_Vehicle_Sales_Counts_by_Month_for_Calendar_Year_2002_through_December_2023.csv"
df = pd.read_csv(url)
df


# In[41]:


df['Date'] = pd.to_datetime(df['Year '].astype(str) + '-' + df['Month '], format='%Y-%b')
df.drop(columns=['Year ', 'Month '], inplace=True)
df.set_index('Date', inplace=True)

#Month start freq
df.index.freq = 'MS' 
df


# In[42]:


url2 = r"C:\Users\urvi9\Annual_Gross_Domestic_Product_Maryland.csv"
df_supporting = pd.read_csv(url2)
df_supporting


# In[43]:


df_supporting['Date'] = pd.to_datetime(df_supporting['Date'])
df_supporting


# In[44]:


merged_df = pd.merge(df, df_supporting, on='Date', how='left')
merged_df.set_index('Date', inplace=True)
merged_df 


# In[45]:


train_data = merged_df.iloc[:-12] 
test_data = merged_df.iloc[-12:]

model_sales_new = SARIMAX(train_data['Total Sales New'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_sales_used = SARIMAX(train_data['Total Sales Used'], order=(1,1,1), seasonal_order=(1,1,1,12))
results_sales_new = model_sales_new.fit()
results_sales_used = model_sales_used.fit()
forecast_sales_new = results_sales_new.get_forecast(steps=12)
forecast_sales_used = results_sales_used.get_forecast(steps=12)

predicted_sales_new = forecast_sales_new.predicted_mean
predicted_sales_used = forecast_sales_used.predicted_mean

plt.figure(figsize=(12,5))
plt.plot(train_data.index, train_data['Total Sales New'], label='Train Data')
plt.plot(test_data.index, test_data['Total Sales New'], label='Test Data')
plt.plot(test_data.index, predicted_sales_new, label='Forecasted Sales New')
plt.legend()
plt.title('Total Sales New Forecast')
plt.show()

plt.figure(figsize=(12,5))
plt.plot(train_data.index, train_data['Total Sales Used'], label='Train Data')
plt.plot(test_data.index, test_data['Total Sales Used'], label='Test Data')
plt.plot(test_data.index, predicted_sales_used, label='Forecasted Sales Used')
plt.legend()
plt.title('Total Sales Used Forecast')
plt.show()


# In[46]:


steps = 60 #(60 months)
forecast_sales_new_extended = results_sales_new.get_forecast(steps=steps)
forecast_sales_used_extended = results_sales_used.get_forecast(steps=steps)
predicted_sales_new_extended = forecast_sales_new_extended.predicted_mean
predicted_sales_used_extended = forecast_sales_used_extended.predicted_mean

plt.figure(figsize=(11,5))
plt.plot(merged_df.index, merged_df['Total Sales New'], label='Actual Sales New',color='orange')
plt.plot(predicted_sales_new_extended.index, predicted_sales_new_extended, label='Forecasted Sales New',color='lightgreen')
plt.legend()
plt.title('Extended Forecast: Total Sales New')
plt.show()

plt.figure(figsize=(11,5))
plt.plot(merged_df.index, merged_df['Total Sales Used'], label='Actual Sales Used',color='orange')
plt.plot(predicted_sales_used_extended.index, predicted_sales_used_extended, label='Forecasted Sales Used',color='lightgreen')
plt.legend()
plt.title('Extended Forecast: Total Sales Used')
plt.show()


# In[50]:


mae_new = mean_absolute_error(test_data['Total Sales New'], predicted_sales_new)
mse_new = mean_squared_error(test_data['Total Sales New'], predicted_sales_new)
print("MAE for Total Sales New:", mae_new)
print("MSE for Total Sales New:", mse_new)
print()
mae_used = mean_absolute_error(test_data['Total Sales Used'], predicted_sales_used)
mse_used = mean_squared_error(test_data['Total Sales Used'], predicted_sales_used)
print("MAE for Total Sales Used:", mae_used)
print("MSE for Total Sales Used:", mse_used)


# In[ ]:




