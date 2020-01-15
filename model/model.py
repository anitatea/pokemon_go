import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
import catboost as cb

import pickle

df = pd.read_csv('data/pokemon_go.csv')
df['city'].unique()
df = df[df['city']=='Toronto']
df
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
df

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
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

select = SelectPercentile(percentile=40)
select.fit(Z_train, y_train)
Z_train = select.transform(Z_train)
Z_test = select.transform(Z_test)


model = LogisticRegression()
model.fit(Z_train, y_train)
model.score(Z_test, y_test)

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))

df
# df1 = pd.read_csv('data/poke.csv')
# l = list(model.predict_proba(Z_test)[8])
# df2 = pd.DataFrame(l,columns=['prob'])
# df3 = pd.concat([df1,df2],axis=1)
# list(df3.sort_values(by=['prob'], ascending=False).head(10)['pokemon'])
# df3.sort_values(by=['prob'], ascending=False).head(10)

#  API key
