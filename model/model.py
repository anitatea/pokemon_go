import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
import catboost as cb

import pickle


df = pd.read_csv('data/pokemon_go.csv')

target = 'pokedex_id'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
     # ('latitude',[CategoricalImputer(), LabelBinarizer()]),
     # ('longitude',[CategoricalImputer(), LabelBinarizer()]),
     # (['local_time'],[SimpleImputer(), StandardScaler()]),
     (['close_to_water'], StandardScaler()),
     ('city',LabelBinarizer()),
     ('weather',LabelBinarizer()),
     (['temperature'],StandardScaler()),
     (['population_density'], StandardScaler()),
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

# select = SelectPercentile(percentile=40)
# select.fit(Z_train, y_train)
# Z_train = select.transform(Z_train)
# Z_test = select.transform(Z_test)


# model = KNeighborsClassifier(n_neighbors=4)
# model.fit(Z_train, y_train)
# pd.DataFrame({
#     'ytrue': y_test,
#     'yhat': model.predict(Z_test)
# })
#
# model.score(Z_train, y_train), model.score(Z_test, y_test)
#
#
# model.predict_proba(Z_test)


model = LogisticRegression(solver='lbfgs')
model.fit(Z_train, y_train)
predict = model.predict_proba(Z_test)

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))










X_train.sample().to_dict(orient='list')

new = pd.DataFrame({

 # 'latitude': [34.010027],
 # 'longitude': [-118.49583899999999],
 # 'local_time': ['2016-09-02T22:14:14'],
 'close_to_water': [True],
 'city': ['Los_Angeles'],
 'weather': ['Clear'],
 'temperature': [19.7],
 'population_density': [4188.391]
})


y_train.unique().shape
pipe.predict_proba(new)[0].shape

prediction = float(pipe.predict(new)[0])


l = list(df['pokedex_id'].unique())
l.sort()
l






# pd.DataFrame({
#     'ytrue': y_test,
#     'yhat': model.predict(Z_test)
# })


# model = cb.CatBoostClassifier(
#     iterations=100,
#     learning_rate=0.5,
# )
#
# model.fit(
#     Z_train, y_train,
#     eval_set=(Z_test, y_test),
#     verbose=False,
#     plot=False,
# )

# model.score(Z_test, y_test)


# pipe = make_pipeline(mapper, model)
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)
# pickle.dump(pipe, open('pipe.pkl', 'wb'))
