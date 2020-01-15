import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
import requests

app = Flask(__name__)
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
df1 = pd.read_csv('data/poke.csv')

# adding favicon
@app.route('/poke.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                               'poke.ico', mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

def weather(lat,long):
    weather_codes =    {
    '200' : 'HeavyRain' ,
    '201' : 'HeavyRain' ,
    '202' : 'HeavyRain' ,
    '210' : 'HeavyRain' ,
    '211' : 'HeavyRain' ,
    '212' : 'HeavyRain' ,
    '221' : 'HeavyRain' ,
    '230' : 'HeavyRain' ,
    '231' : 'HeavyRain' ,
    '232' : 'HeavyRain' ,
    '300' : 'Humid',
    '301' : 'Drizzle',
    '302' : 'DrizzleandBreezy',
    '310' : 'BreezyandOvercast',
    '311' : 'BreezyandPartlyCloudy',
    '312' : 'LightRainandBreezy',
    '313' : 'DrizzleandBreezy',
    '314' : 'HumidandPartlyCloudy',
    '321' : 'LightRainandBreezy',
    '500' : 'LightRain',
    '501' : 'Rain',
    '502' : 'HeavyRain',
    '503' : 'HeavyRain',
    '504' : 'HeavyRain',
    '511' : 'HeavyRain',
    '520' : 'RainandWindy' ,
    '521' : 'RainandWindy' ,
    '522' : 'Windy',
    '531' : 'Windy',
    '600' : 'Breezy',
    '601' : 'HeavyRain',
    '602' : 'HeavyRain',
    '611' : 'HeavyRain',
    '612' : 'HeavyRain',
    '613' : 'HeavyRain',
    '615' : 'HeavyRain',
    '616' : 'HeavyRain',
    '620' : 'HeavyRain',
    '621' : 'HeavyRain',
    '622' : 'DangerouslyWindy',
    '701' : 'WindyandFoggy',
    '711' : 'WindyandFoggy',
    '721' : 'HumidandOvercast',
    '731' : 'Dry',
    '741' : 'Foggy',
    '751' : 'DryandMostlyCloudy',
    '761' : 'WindyandPartlyCloudy',
    '762' : 'DryandPartlyCloudy',
    '771' : 'HumidandOvercast',
    '781' : 'DangerouslyWindy',
    '800' : 'Clear' ,
    '801' : 'PartlyCloudy',
    '802' : 'PartlyCloudy',
    '803' : 'MostlyCloudy',
    '804' : 'Overcast'
    }
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
    ip =  request.headers.get('X-Forwarded-For', request.remote_addr) #request.environ['REMOTE_ADDR'] #'8.8.8.8' #request.remote_addr
    base_url = "https://api.ipgeolocation.io/astronomy?apiKey="
    key = "2e571f58a33c4c92b5b9e83a95d4d554"
    complete_url = base_url + key + "&ip=" + str(ip) + "&lang=en"
    complete_url
    response = requests.get(complete_url)
    x = response.json()
    city = x['location']['city']
    latitude_ip = x['location']['latitude']
    longitude_ip = x['location']['longitude']
    return [city, latitude_ip, longitude_ip]

@app.route('/result', methods=['POST','GET'])
def predict():
    city, latitude, longitude = get_my_ip()
    weather_location, temperature = weather(latitude, longitude)
    if request.method == 'POST':
        result = request.form
    new = pd.DataFrame({ # should be based on the options of the model
         'close_to_water': [True],
         'city': [city],
         'weather': [weather_location],
         'temperature': [temperature],
         'population_density': [result.get('population_density')]
    })
    prediction = pipe.predict(new)[0]
    return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True) #(debug=True), remove this when everything has been built
