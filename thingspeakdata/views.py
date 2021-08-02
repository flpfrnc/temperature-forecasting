from django.shortcuts import render
import requests
import json
import datetime
from dateutil import tz
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import schedule
import time


def dashboard(request):    
    json_data = requests.get('https://thingspeak.com/channels/196384/feed.json').text
    json_loaded = json.loads(json_data)
    local_zone = tz.gettz("America/Fortaleza")
    context = {}

    dados = json_loaded["feeds"]
    leitura = list()
    outdoor_temp = list()
    temp = list()
    air_pressure = list()
    humidity = list()

    for x in range(len(dados)):
        date = dados[x]["created_at"]
        value_ot = dados[x]["field1"]
        value_t = dados[x]["field2"]
        value_ap = dados[x]["field3"]
        value_h = dados[x]["field4"]

        formatted_date = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").astimezone(local_zone).strftime('%d-%m %H:%M')
        leitura.append(formatted_date)
        outdoor_temp.append(value_ot)
        temp.append(value_t)
        air_pressure.append(value_ap)
        humidity.append(value_h)
    
    context = { "data" :  leitura, "outdoor": str(outdoor_temp),  "temperature" : str(temp), "air_pressure" : str(air_pressure), "humidity": str(humidity) }      
    request.session["context"] = context    
    return render(request, 'pages/dashboard.html', {'contextos' : context})


def month2seasons(x):
    if x in [12, 1, 2]:
        season = 'Winter'
    elif x in [3, 4, 5]:
        season = 'Summer'
    elif x in [6, 7, 8, 9]:
        season = 'Monsoon'
    elif x in [10, 11]:
        season = 'Post_Monsoon'
    return season



def hours2timing(x):
    if x in [22,23,0,1,2,3]:
        timing = 'Night'
    elif x in range(4, 12):
        timing = 'Morning'
    elif x in range(12, 17):
        timing = 'Afternoon'
    elif x in range(17, 22):
        timing = 'Evening'
    else:
        timing = 'X'
    return timing



def temperature(request):
    context = request.session.get('context')
    json_data = requests.get('https://thingspeak.com/channels/196384/feed.json').text
    json_loaded = json.loads(json_data)

    df = pd.DataFrame(json_loaded["feeds"])
    print("head1")
    print(df.head())

    df.drop('field3', axis=1, inplace=True)
    df.drop('field4', axis=1, inplace=True)
    df.drop('field5', axis=1, inplace=True)
    df.drop('field6', axis=1, inplace=True)
    df.drop('field8', axis=1, inplace=True)

    print("head2")
    print(df.head())

    df.rename(columns={'created_at':'date', 'field1':'out_temp', 'field2':'temp', 'entry_id':'id'}, inplace=True)

    print("head3")
    print(df.head())

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%SZ')
    df['year'] = df['date'].apply(lambda x : x.year)
    df['month'] = df['date'].apply(lambda x : x.month)
    df['day'] = df['date'].apply(lambda x : x.day)
    df['weekday'] = df['date'].apply(lambda x : x.day_name())
    df['weekofyear'] = df['date'].apply(lambda x : x.weekofyear)
    df['hour'] = df['date'].apply(lambda x : x.hour)
    df['minute'] = df['date'].apply(lambda x : x.minute)

    print("head4")
    print(df.head())

    df['season'] = df['month'].apply(month2seasons)

    print("head5")
    print(df.head())


    df['timing'] = df['hour'].apply(hours2timing)

    print("head6")
    print(df.head())
    #print(df[df.duplicated()])
    #print(df.loc[df['date']=='2021-07-30 11:46:13', ].sort_values(by='id').head(5))

    month_rd = np.round(df['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100,decimals=1)
    print(month_rd)
    month_rd_bar = hv.Bars(month_rd).opts(color="green")
    month_rd_curve = hv.Curve(month_rd).opts(color="red")
    (month_rd_bar * month_rd_curve).opts(title="Monthly Readings Count", xlabel="Month", ylabel="Percentage", yformatter='%d%%', width=700, height=300,tools=['hover'],show_grid=True)

    return render(request, 'pages/temperature.html', {'contextos' : context})

def humidity(request):
    context = request.session.get('context')
  #  schedule.every(1).minutes.do(read_data)
    return render(request, 'pages/humidity.html', {'contextos' : context})

def out_temp(request):
    context = request.session.get('context')
   # schedule.every(1).minutes.do(read_data)
    return render(request, 'pages/out_temp.html', {'contextos' : context})

def pressure(request):
    context = request.session.get('context')
    #schedule.every(1).minutes.do(read_data)
    return render(request, 'pages/pressure.html', {'contextos' : context})

#while True:
#    schedule.run_pending()
#    time.sleep(1)