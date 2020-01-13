import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        result = request.form
    new = pd.DataFrame({ # should be based on the options of the model
         'close_to_water': result['close_to_water'],
         'city': result['city'],
         'weather': result['weather'],
         'temperature': result['temperature'],
         'population_density': result['population_density']
    })

    prediction = pipe.predict(new)[0]
     # prediction = '${:,.2f}'.format(prediction)
    return render_template('result.html', prediction=prediction)
    # return render_template('result.html', prediction=result['pokedex'])

if __name__ == '__main__':
    app.run(debug=True) #(debug=True), remove this when everything has been built
