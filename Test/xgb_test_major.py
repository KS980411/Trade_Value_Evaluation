import os, sys
import numpy as np
import pandas as pd
import pickle
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

# 모델 로드

xgb_regressor = joblib.load('xgb_regressor.pkl')

# 테스트셋 적합

pred = xgb_regressor.predict(test_x)
rmse = np.sqrt(mean_squared_error(test_y, pred))
print(rmse) # rmse

r2 = r2_score(test_y, pred)
print(r2) # R^2

## 데이터프레임화 및 샘플 저장

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/모델 예측치')

pred_origin, test_y_origin = yscaler.scale_to_origin(pred, test_y)
test_y_origin = test_y_origin.reshape(-1)
pred_origin = pred_origin.reshape(-1)

df = pd.DataFrame({"name" : name, "xgboost 예측치" : np.round(pred_origin,1), "트레이드 가치" : np.round(test_y_origin, 1)})
df.to_csv('xgBoost model_major level_예측치.csv', index = False)


# KBO 데이터셋 적합

## 전처리

os.chdir('/Users/choeunsol/Python/PAINS 방중 프로젝트/최종 분석 데이터셋/인풋 데이터')

kbo = pd.read_excel('dat_final_KBO_major.xlsx', sheet_name = 'dat_final_KBO_major')
kbomajor_dataset = kbo.loc[:, ['name', 'team', 'age', 'control.years', 'war_20', 'war_21', 'war_22', 'pos', 'sal']]

foran_kbo = kbomajor_dataset.loc[:, ['age', 'war_20', 'war_21', 'war_22', 'pos', 'sal']]

# 정규화

salary_scaler_kbo = RobustScaler()

k_salary = foran_kbo['sal'].values.reshape(-1,1)
salary_scaler_kbo.fit(k_salary)

kbomajor_dataset.loc[:, ['sal']] = salary_scaler_kbo.transform(k_salary)

pos_dict = {"1B" : 1, "2B" : 2, "3B" : 3, "C" : 4, "DH" : 5, "OF" : 6, "RP" : 7, "SP" : 8, "SS" : 9, "UTIL" : 10}
afterdata_kbo = kbomajor_dataset['pos'].apply(lambda x : pos_dict[x])
kbomajor_dataset['pos'] = afterdata_kbo

kwar = kbomajor_dataset.loc[:, ['war_20', 'war_21', 'war_22']]
kwar *= (162/144)
kbomajor_dataset.loc[:, ['war_20', 'war_21', 'war_22']] = kwar


# KBO에 적합

kbomajor_dataset.rename(columns = {'war_20' : 'fwar_19', 'war_21' : 'fwar_21', 'war_22' : 'fwar_22', 'sal' : 'salary'}, inplace = True)

name = kbomajor_dataset['name'].values
team = kbomajor_dataset['team'].values
field_value = xgb_regressor.predict(kbomajor_dataset.drop(['name', 'team'], axis = 1))
salary_norm = kbomajor_dataset['salary'].values.reshape(-1,1)
kbo_salary = mlb_salary_scaler.inverse_transform(salary_norm).reshape(-1)

kbo_field_value = mlb_fv_scaler.inverse_transform(field_value.reshape(-1, 1))
kbo_field_value = kbo_field_value.reshape(-1)
kbo_salary[kbo_salary < 0] = 0

trade_value = kbo_field_value - kbo_salary

tradevalue_df = pd.DataFrame({'name' : name, 'team' : team, 'trade_value' : np.round(trade_value,2)})




