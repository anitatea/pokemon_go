import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures, LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
import catboost as cb
from ast import literal_eval


import pickle

df = pd.read_csv('scraped_df2.csv')
df['city'].unique()
df = df[df['city']=='Toronto']

df.loc[:,'google_types'] = df.loc[:,'google_types'].apply(lambda x: literal_eval(x))

df['local_time'] = pd.to_datetime(df['local_time'])
df['day'] = df['local_time'].dt.day_name()
df['close_to_water'] = df['close_to_water'].replace({True:'yes', False:'no'})
df['hour'] = df['local_time'].dt.hour
df['temperature'] = round(df['temperature'],0)
df
target = 'pokedex_id'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
list(df.columns)

mapper = DataFrameMapper([
     ('latitude', None),
     ('longitude',None),
     ('hour', LabelBinarizer()),
     ('day', LabelBinarizer()),
     ('close_to_water', LabelEncoder()),
     # ('city',LabelBinarizer()),
     ('weather',LabelBinarizer()),
     (['temperature'],StandardScaler()),
     (['population_density'], StandardScaler()),
     ('google_types', MultiLabelBinarizer()),
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)
Z_train

# poly = PolynomialFeatures(degree=2, include_bias=False)
# poly.fit(Z_train)
# Z_train_poly = poly.transform(Z_train)
# Z_test_poly = poly.transform(Z_test)

# select = SelectPercentile(percentile=40)
# select.fit(Z_train, y_train)
# Z_train = select.transform(Z_train)
# Z_test = select.transform(Z_test)


model = LogisticRegression()
model.fit(Z_train, y_train)
model.score(Z_test, y_test)


pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))






#  API key
