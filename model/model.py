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

df = pd.read_csv('scraped_df.csv')
df['city'].unique()
df = df[df['city']=='Toronto']

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
     ('accounting',None),
     ('airport',None),
     ('amusement_park',None),
     ('aquarium',None),
     ('art_gallery',None),
     ('atm',None),
     ('bakery',None),
     ('bank',None),
     ('bar',None),
     ('beauty_salon',None),
     ('bicycle_store',None),
     ('book_store',None),
     ('bowling_alley',None),
     ('bus_station',None),
     ('cafe',None),
     ('campground',None),
     ('car_dealer',None),
     ('car_rental',None),
     ('car_repair',None),
     ('car_wash',None),
     ('casino',None),
     ('cemetery',None),
     ('church',None),
     ('city_hall',None),
     ('clothing_store',None),
     ('convenience_store',None),
     ('courthouse',None),
     ('dentist',None),
     ('department_store',None),
     ('doctor',None),
     ('drugstore',None),
     ('electrician',None),
     ('electronics_store',None),
     ('embassy',None),
     ('fire_station',None),
     ('florist',None),
     ('funeral_home',None),
     ('furniture_store',None),
     ('gas_station',None),
     ('grocery_or_supermarket',None),
     ('gym',None),
     ('hair_care',None),
     ('hardware_store',None),
     ('hindu_temple',None),
     ('home_goods_store',None),
     ('hospital',None),
     ('insurance_agency',None),
     ('jewelry_store',None),
     ('laundry',None),
     ('lawyer',None),
     ('library',None),
     ('light_rail_station',None),
     ('liquor_store',None),
     ('local_government_office',None),
     ('locksmith',None),
     ('lodging',None),
     ('meal_delivery',None),
     ('meal_takeaway',None),
     ('mosque',None),
     ('movie_rental',None),
     ('movie_theater',None),
     ('moving_company',None),
     ('museum',None),
     ('night_club',None),
     ('painter',None),
     ('park',None),
     ('parking',None),
     ('pet_store',None),
     ('pharmacy',None),
     ('physiotherapist',None),
     ('plumber',None),
     ('police',None),
     ('post_office',None),
     ('primary_school',None),
     ('real_estate_agency',None),
     ('restaurant',None),
     ('roofing_contractor',None),
     ('rv_park',None),
     ('school',None),
     ('secondary_school',None),
     ('shoe_store',None),
     ('shopping_mall',None),
     ('spa',None),
     ('stadium',None),
     ('storage',None),
     ('store',None),
     ('subway_station',None),
     ('supermarket',None),
     ('synagogue',None),
     ('taxi_stand',None),
     ('tourist_attraction',None),
     ('train_station',None),
     ('transit_station',None),
     ('travel_agency',None),
     ('university',None),
     ('veterinary_care',None),
     ('zoo',None),
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

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

df['pokedex_id'].value_counts()

# pipe = make_pipeline(mapper, model)
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)
# pickle.dump(pipe, open('pipe.pkl', 'wb'))



df 




#  API key
