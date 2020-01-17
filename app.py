from datetime import datetime
import io
import os
import pickle
import time

from flask import Flask, request, render_template, send_from_directory, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import requests

from api_keys import weather_api_key, ip_geolocate, google_key
from app_values import weather_codes

app = Flask(__name__)
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
df1 = pd.read_csv('data/poke.csv')

# adding favicon
@app.route('/poke.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                               'poke.ico', mimetype='image/png')

def weather(lat,long):
    ''' Return weather based on Lat and Long '''
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    key = weather_api_key
    complete_url = base_url + "lat=" + f'{lat}' + "&lon=" + f'{long}' + "&units=metric&APPID=" + key
    response = requests.get(complete_url)
    x = response.json()
    weather_temp = round(x['main']['temp'],1)
    weather_id = x['weather'][0]['id']
    return [ weather_codes[f'{weather_id}'] , weather_temp ]

def get_my_ip():
    ''' Return location and timezone offset based on ip '''
    # if run locally replace ip as string
    ip = '8.8.8.8' #request.headers.get('X-Forwarded-For', request.remote_addr) 
    base_url = "https://api.ipgeolocation.io/astronomy?apiKey="
    key = ip_geolocate
    complete_url = base_url + key + "&ip=" + str(ip) + "&lang=en"
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
    ''' Return google place types within 100m of Lat and Long '''
    base_url = "https://maps.googleapis.com/maps/api/place/search/json?location="
    key = google_key
    complete_url = base_url +  f'{lat}' + "," + f'{long}' + "&radius=100&key=" + key
    response = requests.get(complete_url)
    x = response.json()
    y = x['results']
    types = set([])
    for i in y:
        for h in i['types']:
            types.add(h)
    return list(types)

def predict_poke(latitude=0,longitude=0, local_timezone=0):
    ''' Return dictionary of pokemon id's and names '''
    if latitude==0 and longitude==0 and local_timezone==0:
        city, latitude, longitude, local_timezone = get_my_ip()
    weather_location, temperature = weather(latitude, longitude)
    local_hour = int(time.strftime("%H", time.gmtime())) + local_timezone
    local_day = str(time.strftime("%A", time.gmtime()))
    place_list = scrape_place(latitude,longitude)
    new = pd.DataFrame({
            'latitude' : [latitude],
            'longitude' : [longitude],
            'hour' : [local_hour],
            'day' : [local_day],
            'close_to_water': ['yes'],
            'weather': [weather_location],
            'temperature': [temperature],
            'google_types': [place_list],
            'population_density': 4000
    })
    l = list(pipe.predict_proba(new))
    zip_list = list(zip(pipe.classes_,l[0]))
    df1 = pd.DataFrame(zip_list, columns=['id','prob'])
    df2 = pd.read_csv('data/poke.csv')
    new_list = []
    for row in range(len(df1)):
        for i in range(len(df2)):
            if df1['id'][row] == df2['id'][i]:
                new_list.append(df2['pokemon'][i])
        else:
            pass
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
    proba_dict = predict_poke(latitude+.06, longitude, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/south', methods=['POST','GET'])
def south():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude-.06, longitude, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/east', methods=['POST','GET'])
def east():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude, longitude-.06, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/west', methods=['POST','GET'])
def west():
    city, latitude, longitude, local_timezone = get_my_ip()
    proba_dict = predict_poke(latitude, longitude+.06, local_timezone)
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])


@app.route('/result', methods=['POST','GET'])
def manual_result():
    if request.method == 'POST':
        result = request.form
    place_list = scrape_place(result.get('latitude'),result.get('longitude'))
    new = pd.DataFrame({
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
    df2 = pd.read_csv('data/poke.csv')
    new_list = []
    for row in range(len(df1)):
        for i in range(len(df2)):
            if df1['id'][row] == df2['id'][i]:
                new_list.append(df2['pokemon'][i])
        else:
            pass
    df1['pokemon'] = new_list
    out = df1.sort_values(by=['prob'], ascending=False)[['id','pokemon']]
    proba_dict = out.reset_index(drop=True).to_dict(orient='dict')
    return render_template('index.html',  gif=proba_dict['id'], names=proba_dict['pokemon'])

@app.route('/graph', methods=['POST','GET'])
def graph_poke(latitude=0,longitude=0, local_timezone=0):
    ''' Return graph of pokemon probs '''
    if latitude==0 and longitude==0 and local_timezone==0:
        city, latitude, longitude, local_timezone = get_my_ip()
    weather_location, temperature = weather(latitude, longitude)
    local_hour = int(time.strftime("%H", time.gmtime())) + local_timezone
    local_day = str(time.strftime("%A", time.gmtime()))
    place_list = scrape_place(latitude,longitude)
    new = pd.DataFrame({
            'latitude' : [latitude],
            'longitude' : [longitude],
            'hour' : [local_hour],
            'day' : [local_day],
            'close_to_water': ['yes'],
            'weather': [weather_location],
            'temperature': [temperature],
            'google_types': [place_list],
            'population_density': 4000
    })
    l = list(pipe.predict_proba(new))
    zip_list = list(zip(pipe.classes_,l[0]))
    df1 = pd.DataFrame(zip_list, columns=['id','prob'])
    df2 = pd.read_csv('data/poke.csv')
    new_list = []
    for row in range(len(df1)):
        for i in range(len(df2)):
            if df1['id'][row] == df2['id'][i]:
                new_list.append(df2['pokemon'][i])
        else:
            pass
    df1['pokemon'] = new_list
    df2 = df1.sort_values(by=['prob'], ascending=False).head(25).sort_values(by=['prob'], ascending=True)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = df2['prob']
    ys = df2['pokemon']
    axis.barh(ys, xs)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')




@app.route('/manual')
def manual_entry():
    return render_template('manual.html')

if __name__ == '__main__':
    app.run(debug=True)
