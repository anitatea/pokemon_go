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
    ip =  '8.8.8.8'#request.headers.get('X-Forwarded-For', request.remote_addr) 
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

@app.route('/', methods=['POST','GET'])
def index():
    city, latitude, longitude = get_my_ip()
    weather_location, temperature = weather(latitude, longitude)
    local_hour = int(time.strftime("%H", time.gmtime())) #+ offset
    local_hour
    local_day = int(time.strftime("%-d", time.gmtime()))
    local_day
    if request.method == 'POST':
        result = request.form
    new = pd.DataFrame({ # should be based on the options of the model
         'latitude' : [latitude],
         'longitude' : [longitude],
         'hour' : [local_hour],
         'day' : ['Monday'],
         'close_to_water': ['yes'],
         'weather': [weather_location],
         'temperature': [temperature],
         'population_density': 4188.391 # [result.get('population_density')]
    })

    l = pipe.predict_proba(new)
    df1 = pd.read_csv('data/poke.csv')
    df2 = pd.DataFrame(l.reshape(-1,1), columns=['prob'])
    df2
    df3 = pd.concat([df1,df2],axis=1)
    out = df3.sort_values(by=['prob'], ascending=True).head(10)[['id','pokemon']].reset_index()
    #out2 = out.to_dict(orient='split')

    poke_0 = out['pokemon'][0]
    gif_0 = out['id'][0]
    poke_1 = out['pokemon'][1]
    gif_1 = out['id'][1]
    poke_2 = out['pokemon'][2]
    gif_2 = out['id'][2]
    poke_3 = out['pokemon'][3]
    gif_3 = out['id'][3]
    poke_4 = out['pokemon'][4]
    gif_4 = out['id'][4]
    poke_5 = out['pokemon'][5]
    gif_5 = out['id'][5]
    poke_6 = out['pokemon'][6]
    gif_6 = out['id'][6]
    poke_7 = out['pokemon'][7]
    gif_7 = out['id'][7]
    poke_8 = out['pokemon'][8]
    gif_8 = out['id'][8]
    poke_9 = out['pokemon'][9]
    gif_9 = out['id'][9]



    return render_template('result_test.html',  poke_0=poke_0,gif_0=gif_0,poke_1=poke_1,gif_1=gif_1,poke_2=poke_2,gif_2=gif_2,poke_3=poke_3,gif_3=gif_3,poke_4=poke_4,gif_4=gif_4,poke_5=poke_5,gif_5=gif_5,poke_6=poke_6,gif_6=gif_6,poke_7=poke_7,gif_7=gif_7,poke_8=poke_8,gif_8=gif_8,poke_9=poke_9,gif_9=gif_9)
  #  def render_sidebar_template(tmpl_name, **kwargs):
  #  (var1, var2, var3) = generate_sidebar_data()
  #  return render_template(tmpl_name, var1=var1, var2=var2, var3=var3, **kwargs)
# df1 = pd.read_csv('data/poke.csv')
# l = list(model.predict_proba(Z_test)[8])
# df2 = pd.DataFrame(l,columns=['prob'])
# df3 = pd.concat([df1,df2],axis=1)
# list(df3.sort_values(by=['prob'], ascending=False).head(10)['pokemon'])
# df3.sort_values(by=['prob'], ascending=False).head(10)





@app.route('/result', methods=['POST','GET'])
def manual_result():
        if request.method == 'POST':
            result = request.form
        new = pd.DataFrame({ # should be based on the options of the model
             'latitude' : [result.get('latitude')],
             'longitude' : [result.get('longitude')],
             'hour' : [result.get('hour')],
             'day' : [result.get('day')],
             'close_to_water': [result.get('close_to_water')],
             'weather': [result.get('weather')],
             'temperature': [result.get('temperature')],
             'population_density': [result.get('population_density')]
        })
        l = list(pipe.predict_proba(new)[0])
        df2 = pd.DataFrame(l,columns=['prob'])
        df3 = pd.concat([df1,df2],axis=1)
        pred1 = list(df3.sort_values(by=['prob'], ascending=False).head(10)['pokemon'])
        return render_template('index.html', prediction=pred1)

@app.route('/manual')
def manual_entry():
    return render_template('manual.html')

if __name__ == '__main__':
    app.run(debug=True) #(debug=True), remove this when everything has been built
