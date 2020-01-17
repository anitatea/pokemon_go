import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
import requests
from app_values import weather_codes
from datetime import datetime
import time

app = Flask(__name__)
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
df1 = pd.read_csv('data/poke.csv')

# adding favicon
@app.route('/poke.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                               'poke.ico', mimetype='image/png')

def weather(lat,long):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    key = "d7faea5d1fb22532b1fca3da90745d37"
    complete_url = base_url + "lat=" + f'{lat}' + "&lon=" + f'{long}' + "&units=metric&APPID=" + key
    response = requests.get(complete_url)
    complete_url
    x = response.json()
    weather_temp = round(x['main']['temp'],1)
    weather_id = x['weather'][0]['id']
    return [ weather_codes[f'{weather_id}'] , weather_temp ]

def get_my_ip():
    ip =  '72.143.53.170'#request.headers.get('X-Forwarded-For', request.remote_addr) 
    base_url = "https://api.ipgeolocation.io/astronomy?apiKey="
    key = "2e571f58a33c4c92b5b9e83a95d4d554"
    complete_url = base_url + key + "&ip=" + str(ip) + "&lang=en"
    complete_url
    response = requests.get(complete_url)
    x = response.json()
    city = x['location']['city']
    latitude_ip = x['location']['latitude']
    longitude_ip = x['location']['longitude']
    base_url_2 = "https://api.ipgeolocation.io/timezone?apiKey=" 
    complete_url_2 = base_url_2 + key + "&ip=" + str(ip) + "&lang=en"
    response = requests.get(complete_url_2)
    x = response.json()
    local_timezone = x['timezone_offset']
    return [city, latitude_ip, longitude_ip, local_timezone]

def scrape_place(lat, long):
    base_url = "https://maps.googleapis.com/maps/api/place/search/json?location="
    key = "AIzaSyBp9botUwSL1BWEJuLrccoWUtE2nJED9qs"
    complete_url = base_url +  f'{lat}' + "," + f'{long}' + "&radius=100&key=" + key
    response = requests.get(complete_url)
    x = response.json()
    y = x['results']
    y
    types = set([])
    for i in y:
        for h in i['types']:
            types.add(h)
    return list(types)

def predict_poke(latitude=0,longitude=0, local_timezone=0):
    if latitude==0 and longitude==0 and local_timezone==0:
        city, latitude, longitude, local_timezone = get_my_ip()

    weather_location, temperature = weather(latitude, longitude)
    local_hour = int(time.strftime("%H", time.gmtime())) + local_timezone
    local_day = str(time.strftime("%A", time.gmtime()))
    place_list = scrape_place(latitude,longitude)
    new = pd.DataFrame({ # should be based on the options of the model
            'latitude' : [latitude],
            'longitude' : [longitude],
            'hour' : [local_hour],
            'day' : [local_day],
            'close_to_water': ['yes'],
            'weather': [weather_location],
            'temperature': [temperature],
            'google_types': [place_list],
            'population_density': 4188.391 # [result.get('population_density')]
    })
    l = list(pipe.predict_proba(new))
    zip_list = list(zip(pipe.classes_,l[0]))
    df1 = pd.DataFrame(zip_list, columns=['id','prob'])
    # Reading in a DataFrame that has poke ids with there pokemon names
    df2 = pd.read_csv('data/poke.csv')
    # matching the ids then appending the name into an empty list
    new_list = []
    for row in range(len(df1)):
        for i in range(len(df2)):
            if df1['id'][row] == df2['id'][i]:
                new_list.append(df2['pokemon'][i])
        else:
            pass
    # putting the list of names into df1
    df1['pokemon'] = new_list
    out = df1.sort_values(by=['prob'], ascending=False)[['id','pokemon']]
    out2 = out.reset_index(drop=True).to_dict(orient='dict')
    return out2

@app.route('/', methods=['POST','GET'])
def index():
    proba_dict = predict_poke() 
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/north', methods=['POST','GET'])
def north():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude+.3, longitude, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/south', methods=['POST','GET'])
def south():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude-.3, longitude, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/east', methods=['POST','GET'])
def east():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude, longitude-.3, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/west', methods=['POST','GET'])
def west():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude, longitude+.3, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])


@app.route('/result', methods=['POST','GET'])
def manual_result():
    
    if request.method == 'POST':
        result = request.form
    place_list = scrape_place(result.get('latitude'),result.get('longitude'))    
    new = pd.DataFrame({ # should be based on the options of the model
            'latitude' : [result.get('latitude')],
            'longitude' : [result.get('longitude')],
            'hour' : [result.get('hour')],
            'day' : [result.get('day')],
            'close_to_water': [result.get('close_to_water')],
            'weather': [result.get('weather')],
            'temperature': [result.get('temperature')],
            'population_density': [result.get('population_density')],
            'google_types': [place_list]
    })
    l = list(pipe.predict_proba(new))
    zip_list = list(zip(pipe.classes_,l[0]))
    df1 = pd.DataFrame(zip_list, columns=['id','prob'])
    # Reading in a DataFrame that has poke ids with there pokemon names
    df2 = pd.read_csv('data/poke.csv')
    # matching the ids then appending the name into an empty list
    new_list = []
    for row in range(len(df1)):
        for i in range(len(df2)):
            if df1['id'][row] == df2['id'][i]:
                new_list.append(df2['pokemon'][i])
        else:
            pass
    # putting the list of names into df1
    df1['pokemon'] = new_list
    out = df1.sort_values(by=['prob'], ascending=False)[['id','pokemon']]
    proba_dict = out.reset_index(drop=True).to_dict(orient='dict')
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/manual')
def manual_entry():
    return render_template('manual.html')

if __name__ == '__main__':
    app.run(debug=True) #(debug=True), remove this when everything has been built
