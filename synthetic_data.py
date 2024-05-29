import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import timesynth as ts

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer

df=pd.read_csv('Airwave_OG.csv')

df.interpolate(inplace=True)

data = df.copy()

data = data.drop([
    'SIP Dropped Calls',
    'Cat M1 Bearer Drop pct',
    'Pct CA ScheduledUE with 0 EScell DL',
    'Pct CA ScheduledUE with 1 EScell DL',
    'Pct CA ScheduledUE with 2 EScell DL',
    'Pct CA ScheduledUE with 3 EScell DL',
    'SIP Calls with a Leg',
    'Pct CA ScheduledUE with 4 EScell DL',
    'Cat M1 Bearer Setup Failure pct',
    '_80th_percentile_traffic',
    'SIP DC%',
    'Pct CA ScheduledUE with 1 Scell UL',
    'Pct CA ScheduledUE with 2 Scell UL',
    'Pct CA ScheduledUE with 3 Scell UL',
    'HO_fail_PCT_InterFreq',
    'day'
],
          axis = 1)

columns_to_iterate = []

for i in data.columns:
    if i == 'Avg_Connected_UEs':
        continue
    if df[i].dtypes != 'object':
        correlation = df[[i, 'Avg_Connected_UEs']].corr().iloc[0, 1]
        if not pd.isna(correlation):
            columns_to_iterate.append(i)


for i in trange(48):

    time_sampler_pp = ts.TimeSampler(stop_time=72)
    irregular_time_samples_pp = time_sampler_pp.sample_irregular_time(resolution=0.1, keep_percentage=100)
    pseudo_periodic = ts.signals.PseudoPeriodic(frequency=1.32, freqSD=0.001, ampSD=0.4)
    timeseries_pp = ts.TimeSeries(pseudo_periodic)
    samples_pp, signals_pp, errors_pp = timeseries_pp.sample(irregular_time_samples_pp)

    samples_pp = (abs(samples_pp) * 8) + 1
    
    X_unseen = samples_pp

    X = df['Avg_Connected_UEs']
    predicted_values = {}

    for column in tqdm(columns_to_iterate, leave=False):
        y = df[column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Splitting data
        models = {
            'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100, 150]}),
            'XGBoost': (xgb.XGBRegressor(), {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5, 7]})
        }

        best_score = -float('inf')
        best_model = None
        best_predictions = None

        for name, (model, parameters) in models.items():
            grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train.values.reshape(-1, 1), y_train)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_predictions = grid_search.predict(X_test.values.reshape(-1, 1))
            X_unseen_reshaped =np.array(X_unseen).reshape(-1,1)

            predicted_values[column] = best_model.predict(X_unseen_reshaped)

        predicted_values['Avg_Connected_UEs'] = samples_pp

        values = pd.DataFrame(predicted_values)
        values.to_csv(f'/Data/values_{i+1}.csv')
