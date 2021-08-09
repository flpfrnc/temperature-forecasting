from django.shortcuts import render
from pandas.core.frame import DataFrame
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
import threading


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
    
    return render(request, 'pages/temperature.html', {'contextos' : context})

def humidity(request):
    context = request.session.get('context')
    return render(request, 'pages/humidity.html', {'contextos' : context})

def out_temp(request):
    context = request.session.get('context')
    return render(request, 'pages/out_temp.html', {'contextos' : context})

def pressure(request):
    context = request.session.get('context')
    return render(request, 'pages/pressure.html', {'contextos' : context})


def predict():
    json_data = requests.get('https://thingspeak.com/channels/196384/feed.json').text
    json_loaded = json.loads(json_data)

    df1 = pd.DataFrame(json_loaded["feeds"])
    df2 = pd.DataFrame(json_loaded["feeds"])


    df1.drop('field1', axis=1, inplace=True)
    df1.drop('field3', axis=1, inplace=True)
    df1.drop('field4', axis=1, inplace=True)
    df1.drop('field5', axis=1, inplace=True)
    df1.drop('field6', axis=1, inplace=True)
    df1.drop('field8', axis=1, inplace=True)

    df2.drop('field2', axis=1, inplace=True)
    df2.drop('field3', axis=1, inplace=True)
    df2.drop('field4', axis=1, inplace=True)
    df2.drop('field5', axis=1, inplace=True)
    df2.drop('field6', axis=1, inplace=True)
    df2.drop('field8', axis=1, inplace=True)

    df1.rename(columns={'created_at':'date', 'field2':'temp', 'entry_id':'id'}, inplace=True)
    df2.rename(columns={'created_at':'date', 'field1':'temp', 'entry_id':'id'}, inplace=True)
    df1["place"] = 'In'
    df2["place"] = 'Out'

    df = DataFrame()
    df = pd.concat([df1, df2], axis=0)

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%SZ')
    df['year'] = df['date'].apply(lambda x : x.year)
    df['month'] = df['date'].apply(lambda x : x.month)
    df['day'] = df['date'].apply(lambda x : x.day)
    df['weekday'] = df['date'].apply(lambda x : x.day_name())
    df['weekofyear'] = df['date'].apply(lambda x : x.weekofyear)
    df['hour'] = df['date'].apply(lambda x : x.hour)
    df['minute'] = df['date'].apply(lambda x : x.minute)


    df['season'] = df['month'].apply(month2seasons)

    df['timing'] = df['hour'].apply(hours2timing)


    df[df.duplicated()]
    df.drop_duplicates(inplace=True)
    df[df.duplicated()]
    
    #print(df[df.duplicated()])
    #print(df.loc[df['date']=='2021-07-30 11:46:13', ].sort_values(by='id').head(5))

    month_rd = np.round(df['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100,decimals=1)
    month_rd_bar = hv.Bars(month_rd).opts(color="green")
    month_rd_curve = hv.Curve(month_rd).opts(color="red")
    (month_rd_bar * month_rd_curve).opts(title="Monthly Readings Count", xlabel="Month", ylabel="Percentage", yformatter='%d%%', width=700, height=300,tools=['hover'],show_grid=True)


    hv.Distribution(df['temp']).opts(title="Temperature Distribution", color="green", xlabel="Temperature", ylabel="Density")\
                            .opts(opts.Distribution(width=700, height=300,tools=['hover'],show_grid=True))

    pl_cnt = np.round(df['place'].value_counts(normalize=True) * 100)
    hv.Bars(pl_cnt).opts(title="Readings Place Count", color="green", xlabel="Places", ylabel="Percentage", yformatter='%d%%')\
                    .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))

    season_cnt = np.round(df['season'].value_counts(normalize=True) * 100)
    hv.Bars(season_cnt).opts(title="Season Count", color="green", xlabel="Season", ylabel="Percentage", yformatter='%d%%')\
                    .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


    timing_cnt = np.round(df['timing'].value_counts(normalize=True) * 100)
    hv.Bars(timing_cnt).opts(title="Timing Count", color="green", xlabel="Timing", ylabel="Percentage", yformatter='%d%%')\
                    .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))

    

    in_month = np.round(df[df['place']=='In']['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100, decimals=1)
    out_month = np.round(df[df['place']=='Out']['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100, decimals=1)
    in_out_month = pd.merge(in_month,out_month,right_index=True,left_index=True).rename(columns={'date_x':'In', 'date_y':'Out'})
    in_out_month = pd.melt(in_out_month.reset_index(), ['index']).rename(columns={'index':'Month', 'variable':'Place'})
    hv.Bars(in_out_month, ['Month', 'Place'], 'value').opts(opts.Bars(title="Monthly Readings by Place Count", width=700, height=400,tools=['hover'],show_grid=True, ylabel="Count"))

    

    (hv.Distribution(df[df['place']=='In']['temp'], label='In') * hv.Distribution(df[df['place']=='Out']['temp'], label='Out'))\
                                .opts(title="Temperature by Place Distribution", xlabel="Temperature", ylabel="Density")\
                                .opts(opts.Distribution(width=700, height=300,tools=['hover'],show_grid=True))



    timing_agg = df.groupby('timing').agg({'temp': ['min', 'max']})
    timing_maxmin = pd.merge(timing_agg['temp']['max'],timing_agg['temp']['min'],right_index=True,left_index=True)
    timing_maxmin = pd.melt(timing_maxmin.reset_index(), ['timing']).rename(columns={'timing':'Timing', 'variable':'Max/Min'})
    hv.Bars(timing_maxmin, ['Timing', 'Max/Min'], 'value').opts(title="Temperature by Timing Max/Min", ylabel="Temperature")\
                                                                        .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))



    tsdf = df.drop_duplicates(subset=['date','place']).sort_values('date').reset_index(drop=True)
    tsdf['temp'] = df.groupby(['date','place'])['temp'].mean().values
    tsdf.drop('id', axis=1, inplace=True)
    


    in_month = tsdf[tsdf['place']=='In'].groupby('month').agg({'temp':['mean']})
    in_month.columns = [f"{i[0]}_{i[1]}" for i in in_month.columns]
    out_month = tsdf[tsdf['place']=='Out'].groupby('month').agg({'temp':['mean']})
    out_month.columns = [f"{i[0]}_{i[1]}" for i in out_month.columns]
    hv.Curve(in_month, label='In') * hv.Curve(out_month, label='Out').opts(title="Monthly Temperature Mean", ylabel="Temperature", xlabel='Month')\
                                                                        .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))



    tsdf['hourly'] = tsdf['date'].apply(lambda x : pd.to_datetime(x.strftime('%Y-%m-%d %H:%M')))
    in_hour = tsdf[tsdf['place']=='In'].groupby(['hourly']).agg({'temp':['mean']})
    in_hour.columns = [f"{i[0]}_{i[1]}" for i in in_hour.columns]
    out_hour = tsdf[tsdf['place']=='Out'].groupby(['hourly']).agg({'temp':['mean']})
    out_hour.columns = [f"{i[0]}_{i[1]}" for i in out_hour.columns]
    (hv.Curve(in_hour, label='In') * hv.Curve(out_hour, label='Out')).opts(title="Hourly Temperature Mean", ylabel="Temperature", xlabel='Hour', shared_axes=False)\
                                                                        .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))



    in_wd = tsdf[tsdf['place']=='In'].groupby('weekday').agg({'temp':['mean']})
    in_wd.columns = [f"{i[0]}_{i[1]}" for i in in_wd.columns]
    in_wd['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in in_wd.index]
    in_wd.sort_values('week_num', inplace=True)
    in_wd.drop('week_num', axis=1, inplace=True)
    out_wd = tsdf[tsdf['place']=='Out'].groupby('weekday').agg({'temp':['mean']})
    out_wd.columns = [f"{i[0]}_{i[1]}" for i in out_wd.columns]
    out_wd['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in out_wd.index]
    out_wd.sort_values('week_num', inplace=True)
    out_wd.drop('week_num', axis=1, inplace=True)
    hv.Curve(in_wd, label='In') * hv.Curve(out_wd, label='Out').opts(title="Weekday Temperature Mean", ylabel="Temperature", xlabel='Weekday')\
                                                                        .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))




    in_wof = tsdf[tsdf['place']=='In'].groupby('weekofyear').agg({'temp':['mean']})
    in_wof.columns = [f"{i[0]}_{i[1]}" for i in in_wof.columns]
    out_wof = tsdf[tsdf['place']=='Out'].groupby('weekofyear').agg({'temp':['mean']})
    out_wof.columns = [f"{i[0]}_{i[1]}" for i in out_wof.columns]
    hv.Curve(in_wof, label='In') * hv.Curve(out_wof, label='Out').opts(title="WeekofYear Temperature Mean", ylabel="Temperature", xlabel='WeekofYear')\
                                                                        .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))
    
    
    in_tsdf = tsdf[tsdf['place']=='In'].reset_index(drop=True)
    in_tsdf.index = in_tsdf['date']
    in_all = hv.Curve(in_tsdf['temp']).opts(title="[In] Temperature All", ylabel="Temperature", xlabel='Time', color='red')

    out_tsdf = tsdf[tsdf['place']=='Out'].reset_index(drop=True)
    out_tsdf.index = out_tsdf['date']
    out_all = hv.Curve(out_tsdf['temp']).opts(title="[Out] Temperature All", ylabel="Temperature", xlabel='Time', color='blue')

    in_tsdf_int = in_tsdf['temp'].resample('1min').interpolate(method='nearest')
    in_tsdf_int_all = hv.Curve(in_tsdf_int).opts(title="[In] Temperature All Interpolated with 'nearest'", ylabel="Temperature", xlabel='Time', color='red', fontsize={'title':11})
    out_tsdf_int = out_tsdf['temp'].resample('1min').interpolate(method='nearest')
    out_tsdf_int_all = hv.Curve(out_tsdf_int).opts(title="[Out] Temperature All Interpolated with 'nearest'", ylabel="Temperature", xlabel='Time', color='blue', fontsize={'title':11})

    (in_all + in_tsdf_int_all + out_all + out_tsdf_int_all).opts(opts.Curve(width=400, height=300,tools=['hover'],show_grid=True)).opts(shared_axes=False).cols(2)
        


    in_d_org = hv.Curve(in_hour).opts(title="[In] Daily Temperature Mean", ylabel="Temperature", xlabel='Time', color='red')
    out_d_org = hv.Curve(out_hour).opts(title="[Out] Daily Temperature Mean", ylabel="Temperature", xlabel='Time', color='blue')

    inp_df = pd.DataFrame()
    in_d_inp = in_hour.resample('H').interpolate(method='nearest', order=5)
    out_d_inp = out_hour.resample('H').interpolate(method='nearest', order=5)
    inp_df['In'] = in_d_inp.temp_mean
    inp_df['Out'] = out_d_inp.temp_mean

    in_d_inp_g = hv.Curve(inp_df['In']).opts(title="[In] Daily Temperature Mean Interpolated with 'spline'", ylabel="Temperature", xlabel='Time', color='red', fontsize={'title':10})
    out_d_inp_g = hv.Curve(inp_df['Out']).opts(title="[Out] Daily Temperature Mean Interpolated with 'spline'", ylabel="Temperature", xlabel='Time', color='blue', fontsize={'title':10})

    (in_d_org + in_d_inp_g + out_d_org + out_d_inp_g).opts(opts.Curve(width=400, height=300,tools=['hover'],show_grid=True)).opts(shared_axes=False).cols(2)
    
    
    org_df = inp_df.reset_index()
    org_df['timing'] = org_df['hourly'].apply(lambda x : hours2timing(x.hour))
    org_df = pd.get_dummies(org_df, columns=['timing'])
    
    


    pd.set_option("display.max_rows", None, "display.max_columns", None)
    def run_prophet(place, prediction_periods, plot_comp=True):
        # make dataframe for training
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = pd.date_range(start=org_df['hourly'][0], end=org_df['hourly'][len(org_df)-1], freq='H')
        prophet_df['y'] = org_df[place]
        
        
        # add seasonal information
        if 'timing_Night' in org_df.columns:
            prophet_df['night'] = org_df['timing_Night']
        if 'timing_Evening' in org_df.columns:
            prophet_df['evening'] = org_df['timing_Evening']
        if 'timing_Afternoon' in org_df.columns:
            prophet_df['afternoon'] = org_df['timing_Afternoon']
        if 'timing_Morning' in org_df.columns:
            prophet_df['morning'] = org_df['timing_Morning']
        

        # train model by Prophet
        m = Prophet(changepoint_prior_scale=20, growth='linear', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name='hourly', period=1/24, fourier_order=1)
        
        m.fit(prophet_df)

        # make dataframe for prediction
        future = m.make_future_dataframe(freq='H', periods=24 * prediction_periods)
        # add seasonal information
        future_season = pd.get_dummies(future['ds'].apply(lambda x : hours2timing(x.hour)))
        
        if 'Night' in future_season.columns:
            future['monsoon'] = future_season['Night']
        if 'Evening' in future_season.columns:
            future['evening'] = future_season['Evening']
        if 'Afternoon' in future_season.columns:
            future['afternoon'] = future_season['Afternoon']
        if 'Morning' in future_season.columns:
            future['morning'] = future_season['Morning']

        # predict the future temperature
        prophe_result = m.predict(future)
        # plot prediction
        fig1 = m.plot(prophe_result)
        if place == 'In':
            m.plot(prophe_result).savefig('thingspeakdata/static/images/fig1in.png')
        else:
            m.plot(prophe_result).savefig('thingspeakdata/static/images/fig1out.png')
        ax = fig1.gca()
        ax.set_title(f"{place} Prediction", size=25)
        ax.set_xlabel("Time", size=15)
        ax.set_ylabel("Temperature", size=15)
        a = add_changepoints_to_plot(ax, m, prophe_result)
        print(ax)
        print(ax.plot())
        fig1.show()
        # plot decomposed timse-series components
        if plot_comp:
            fig2 = m.plot_components(prophe_result)
            if place == 'In':
                m.plot_components(prophe_result).savefig('thingspeakdata/static/images/fig2in.png')
            else:
                m.plot_components(prophe_result).savefig('thingspeakdata/static/images/fig2out.png')
            fig2.show()

    run_prophet('In', 30)
    run_prophet('Out', 30)


threading.Timer(5, predict).start()

