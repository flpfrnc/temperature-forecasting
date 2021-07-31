from django.shortcuts import render
import requests
import json
import datetime
from dateutil import tz
import requests

def dashboard(request):
    json_data = requests.get('https://thingspeak.com/channels/196384/feed.json').text
    json_loaded = json.loads(json_data)
    local_zone = tz.gettz("America/Fortaleza")
    context = {}

    dados = json_loaded["feeds"][-500:]
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

