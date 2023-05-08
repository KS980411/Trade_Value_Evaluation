import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import graphviz
import pickle
from sklearn.externals import joblib
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, r2_score
from xgb_tuning_major import xgb_params_tuning

sampler = TPESampler(seed = 0)

# 튜닝

tuned = optuna.create_study(
    study_name = 'xgboost params',
    direction = 'minimize',
    sampler = sampler
)

tuned.optimize(xgb_params_tuning, n_trials = 100)

tuned.best_value # best value 확인

tuned.best_params # best parameters 확인

optuna.visualization.plot_param_importances(tuned) # feature 별 중요도 출력

# train

xgb_regressor = xgb.XGBRegressor(**tuned.best_params)
xgb_regressor.fit(train_x, train_y)

xgb.plot_tree(xgb_regressor,num_trees=1) # 예시 tree 출력
plt.rcParams['figure.figsize'] = [450,400]
plt.show()

# 모델 저장

xgb_major = pickle.dumps(xgb_regressor)
joblib.dump(xgb_major, 'xgb_regressor.pkl')
