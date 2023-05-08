# 라이브러리 로드
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import pickle
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score, mean_squared_error
from scaling import scaled_into_trainset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# 데이터 전처리

## 

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/인풋 데이터')

minor_data = pd.read_csv('MLB 마이너레벨 분석용 데이터셋.csv')
minor_data = minor_data.drop('Unnamed: 0', axis = 1)

## 40인 로스터 여부 원핫인코딩

minor_data['MAJOR_40'] = minor_data.loc[:, 'MAJOR_40'].fillna(0)
minor_with_nona = minor_data.dropna(axis = 0)

## 중복치 처리

minor_with_nona = minor_with_nona.drop_duplicates(['NAME', 'median.trade.value'], keep = False)

## FV 처리
minor_with_nona.groupby('FV').mean().loc[:, 'median.trade.value']

minor_with_nona = minor_with_nona.reset_index().drop('index', axis = 1)
minor_with_nona = minor_with_nona.replace(['35+', '40+', '45+'], [37.5, 42.5, 47.5])
minor_with_nona.groupby('FV').mean().loc[:, 'median.trade.value']

## 범주형 변수 처리

minor_xgboost = minor_with_nona
eta_dict = {2022.0 : 1, 2023.0 : 2, 2024.0 : 3, 2025.0 : 4, 2026.0 : 5, 2027.0 : 6, 2028.0 : 7}
pos_dict = {"1B" : 1, "2B" : 2, "3B" : 3, "C" : 4, "DH" : 5, "OF" : 6, "RP" : 7, "SP" : 8, "SS" : 9, "UTIL" : 10}
eta_real = minor_xgboost['ETA'].apply(lambda x : eta_dict[x])
pos_real = minor_xgboost['POS'].apply(lambda x : pos_dict[x])
minor_xgboost['ETA'] = eta_real
minor_xgboost['POS'] = pos_real

xgdata = minor_xgboost.drop(['NAME', 'MAJOR_40'], axis = 1)

minor_Y = xgdata.loc[:, 'median.trade.value']
minor_X_col = list(xgdata.columns.difference(['median.trade.value']))
minor_X = xgdata[minor_X_col]

minor_X['FV'] = minor_X.loc[:, 'FV'].astype('float64')

## train-test 분리

train_minor_x, test_minor_x, train_minor_y, test_minor_y = train_test_split(minor_X, minor_Y, test_size = 0.2, random_state = 0)
train_minor_x, val_minor_x, train_minor_y, val_minor_y = train_test_split(train_minor_x, train_minor_y, test_size = 1/3, random_state=0)

minor_name = minor_with_nona.iloc[test_minor_y.index].loc[:, 'NAME']

## 정규화

scaletr_minor_x = train_minor_x.loc[:, ['FV']]
scaleval_minor_x = val_minor_x.loc[:, ['FV']]
scaletest_minor_x = test_minor_x.loc[:, ['FV']]

tv_milb_scaler = StandardScaler()
tv_milb_scaler.fit(train_minor_y.values.reshape(-1,1))

minor_y_scaler = scaled_into_trainset(StandardScaler())
train_minor_y, test_minor_y, val_minor_y = minor_y_scaler.y_scaled(train_minor_y, test_minor_y, val_minor_y)

# 파라미터 튜닝

def xgb_params_tuning(trial: Trial) -> float:
    params_xgb = {
        "n_estimators" : trial.suggest_int('n_estimators', 50, 200),
        "max_depth": trial.suggest_int('max_depth', 5, 15),
        'random_state': 0,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.0, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.0, 1.0),
        'learning_rate':trial.suggest_uniform('learning_rate', 0.01, 0.5)
    }
    
    model = xgb.XGBRegressor(**params_xgb)
    model.fit(train_minor_x, train_minor_y, eval_set = [(train_minor_x, train_minor_y), (val_minor_x, val_minor_y)], verbose= False)


    xgb_pred_m = model.predict(val_minor_x)
    rmse = np.sqrt(mean_squared_error(val_minor_y, xgb_pred_m))

    return rmse

sampler = TPESampler(seed = 0)

tuned = optuna.create_study(
    study_name = 'xgboost minor level params',
    direction = 'minimize',
    sampler = sampler
)

tuned.optimize(xgb_params_tuning, n_trials = 400)

# train

import xgboost as xgb

xgb_regressor_minor = xgb.XGBRegressor(**tuned.best_params)
xgb_regressor_minor.fit(train_minor_x, train_minor_y)

pred = xgb_regressor_minor.predict(test_minor_x)
rmse = np.sqrt(mean_squared_error(test_minor_y, pred))
print(rmse) # rmse

r2 = r2_score(test_minor_y, pred)
print(r2) # R^2

## 모델 저장

xgb_minor = pickle.dumps(xgb_regressor_minor)
joblib.dump(xgb_minor, 'xgb_regressor.pkl')

## 예측치 저장
os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/모델 예측치')

pred_origin, real_minor_y = minor_y_scaler.scale_to_origin(pred, test_minor_y)

pred_minor = pred_origin.reshape(-1)
value_minor = real_minor_y.reshape(-1)

df = pd.DataFrame({"name" : minor_name, "predict_value" : np.round(pred_minor,1), "real_value" : np.round(value_minor, 1)})
df.to_csv('xgBoost model_minor level_예측치.csv', index = False)


# KBO 데이터 적합

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/인풋 데이터')

minor_kbo = pd.read_excel('kbo prospect.xlsx')

## 전처리 (군복무 여부 추가)
minor_kbo_name, minor_kbo_team = minor_kbo['name'].values, minor_kbo['team'].values

no_mil = minor_kbo[lambda x : (x.군 == '미필') | (x.군 == '군복무중')]['ETA'] + 1
minor_kbo['ETA'] = no_mil
minor_kbo

## 정규화

minor_kbo_foran = minor_kbo.loc[:, ['age', 'pos', 'score', 'ETA']]
minor_kbo_pos = minor_kbo_foran['pos'].apply(lambda x : pos_dict[x])
minor_kbo_foran['pos'] = minor_kbo_pos

minor_kbo_foran.rename(columns = {'age' : 'AGE', 'pos' : 'POS', 'score' : 'FV'}, inplace = True)
minor_kbo_foran = minor_kbo_foran[['AGE', 'ETA', 'FV', 'POS']]

# predict

minor_trade_value_norm = xgb_regressor_minor.predict(minor_kbo_foran)
minor_trade_value_tf = tv_milb_scaler.inverse_transform(minor_trade_value_norm.reshape(-1,1))
minor_trade_value = minor_trade_value_tf.reshape(-1)

tradevalue_df = pd.DataFrame({'name' : minor_kbo_name, 'team' : minor_kbo_team, 'trade_value' : np.round(minor_trade_value,2)})

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/모델 예측치')
tradevalue_df.to_excel('kbo minor level trade value.xlsx')