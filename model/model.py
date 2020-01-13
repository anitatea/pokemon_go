from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper, CategoricalImputer
import pandas as pd
import pickle

df = pd.read_csv('data/train.csv')

# df = df.dropna()
df['Engine'] = df['Engine'].str.replace(r' CC','').apply(float)
df = df[~df['Power'].str.contains('null')] # remove the 'null bhp'
df['Power'] = df['Power'].str.replace(r' bhp','').apply(float)

# Convert 'Price' from Lahk INR to CAD (as of Jan 11th, 2020)
df['Price'] = df['Price']*1842.83

target = 'Price'
X = df[['Location', 'Year', 'Fuel_Type', 'Transmission', 'Engine', 'Power', 'Seats']]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    ('Location', [CategoricalImputer(), LabelBinarizer()]),
    ('Year', [CategoricalImputer(), LabelBinarizer()]),
    ('Fuel_Type', [CategoricalImputer(), LabelBinarizer()]),
    ('Transmission', [CategoricalImputer(), LabelBinarizer()]),
    (['Engine'], [SimpleImputer(), PolynomialFeatures(include_bias=False), StandardScaler()]),
    (['Power'], [SimpleImputer(), PolynomialFeatures(include_bias=False), StandardScaler()]),
    (['Seats'], [SimpleImputer(), PolynomialFeatures(include_bias=False), StandardScaler()])
])

model = LinearRegression()

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('model/pipe.pkl', 'wb'))

# # load from a model
# del pipe
# pipe = pickle.load(open('model/pipe.pkl', 'rb'))
