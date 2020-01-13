import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
df1 = pd.read_csv('data/poke.csv')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        result = request.form
    new = pd.DataFrame({ # should be based on the options of the model
         'close_to_water': [True],
         'city': [result.get('city')],
         'weather': [result.get('weather')],
         'temperature': [result.get('temperature')],
         'population_density': [result.get('population_density')]
    })

    l = list(model.predict_proba(new))
    df2 = pd.DataFrame(l,columns=['prob'])
    df3 = pd.concat([df1,df2],axis=1)
    df3.sort_values(by=['prob'], ascending=False).head(10)

    return render_template('result.html', prediction=df3[0])

if __name__ == '__main__':
    app.run(debug=True)
