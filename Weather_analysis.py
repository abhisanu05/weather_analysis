#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


data = pd.read_csv("C:/Users/HP/Downloads/archive (2)/DailyDelhiClimateTrain.csv")
data
data.head(10)


# In[3]:


data.describe()


# In[4]:


data.info()


# In[5]:


figure = px.line(data, x ="date",
                y="meantemp", title ="Mean temp in delhi over years")
figure.show()


# In[6]:


figure = px.line(data, x="date", 
                 y="humidity", 
                 title='Humidity in Delhi Over the Years')
figure.show()


# In[7]:


figure = px.line(data, x="date",
                y="wind_speed",
                title ="Wind speed in delhi over years")
figure.show()


# In[8]:


figure = px.scatter(data, x="humidity",
                   y= "meantemp", size="meantemp",
                   trendline="ols",
                   title ="relationship between temperature and humidity")
figure.show()


# In[9]:


data["date"] = pd.to_datetime(data["date"], format = '%Y-%m-%d')
data['year'] = data['date'].dt.year
data["month"] = data["date"].dt.month
print(data.head())


# In[12]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(25, 20))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x='month', y='meantemp', hue='year')
plt.show()


# In[13]:


pip install prophet


# In[14]:


forecast_data = data.rename(columns = {"date": "ds", 
                                       "meantemp": "y"})
print(forecast_data)


# In[15]:


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions)

