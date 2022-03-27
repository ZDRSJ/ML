# 필요 라이브러리 install
! pip install xgboost
! pip install optuna
! pip install lightgbm

# 필요한 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import re
import optuna
from optuna.integration import XGBoostPruningCallback
sns.set_theme(style="darkgrid")

# 데이터 불러오기
df = pd.read_csv('../data/ML_data_final.csv')

df.drop('Unnamed: 0', axis=1, inplace= True)

# train-test분리
cut = int(len(df)*0.8) # traun, valid 80:20 으로 split
train_tmp = df[:cut]
test = df[cut:]
df = train_tmp

# train 데이터셋 분할
train_X = test.drop('price', axis=1)
train_y = test['price']

# 교차 검증 
# 모델 선정 위해 (XGB, LGB) 모델의 성능 비교
'''
 시계열 데이터 - TimeSeriesSplit 기법을 사용
 10개의 폴드로 구분
  1) 과거의 데이터부터 끊어서 학습
  2) 다음 폴드부터 이전 폴드까지 이용하여 반복적으로 교차검증
  3) 미래의 데이터는 절대 학습하지 않으며 무조건 검증으로 사용됨
'''


def RMSE(y, y_pred):
    rmse = mean_squared_error(y, y_pred) ** 0.5 
    return rmse

def rmse_cv(model):
    # cv별로 학습하는 함수
    tscv = TimeSeriesSplit(n_splits=10) # 10개의 폴드로 구분
    rmse_list = []
    model_name = model.__class__.__name__
    for _, (train_index, test_index) in tqdm(enumerate(tscv.split(train_X), start=1), desc=f'{model_name} Cross Validations...', total=10):
        X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
        clf = model.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rmse = RMSE(y_test, pred) 
        rmse_list.append(rmse)
    return model_name, rmse_list

def print_rmse_score(model):
    # cv별 프린팅, 평균 저장
    model_name, score = rmse_cv(model)
    for i, r in enumerate(score, start=1):
        print(f'{i} FOLDS: {model_name} RMSLE: {r:.4f}')
    print(f'\n{model_name} mean RMSLE: {np.mean(score):.4f}')
    print('='*40)
    return model_name, np.mean(score)

# XGB, LGB 모델 설정 
model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=9, min_child_weight=5, gamma=0.1, n_jobs=-1)
model_lgb = lgb.LGBMRegressor(n_estimators=500, max_depth=9, min_child_weight=5, n_jobs=-1)

models = []
scores = []
for model in [model_xgb, model_lgb]:
    model_name, mean_score = print_rmse_score(model)
    models.append(model_name)
    scores.append(mean_score)

# 모델 성능 비교
result_df = pd.DataFrame({'Model': models, 'Score': scores}).reset_index(drop=True)

# 모델 성능 시각화하여 확인
f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation='90')
sns.barplot(x=result_df['Model'], y=result_df['Score'])
plt.xlabel('Models', fontsize=15)
plt.ylabel('Model Performance', fontsize=15)
plt.ylim(0.22, 22000)
plt.title('RMSLE', fontsize=15)
plt.show()

# train, valid split 
cut = int(len(df)*0.8) # traun, valid 80:20 으로 split
train = df[:cut]
valid = df[cut:]

train_X = train.drop('price', axis=1)
train_y = train['price']
valid_X = valid.drop('price', axis=1)
valid_y = valid['price']

# 하이퍼 파라미터 튜닝
# 시간 이슈 때문에 TimeSeriesSplit은 적용하지 않음
from optuna.samplers import TPESampler

sampler = TPESampler(seed=10)

def objective(trial):
    dtrain = lgb.Dataset(train_X, label=train_y)
    dtest = lgb.Dataset(valid_X, label=valid_y)

    param = {
        'objective': 'regression', # 회귀
        'verbose': -1,
        'metric': 'rmse', 
        'max_depth': trial.suggest_int('max_depth',3, 15),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
    }

    model = lgb.LGBMRegressor(**param)
    lgb_model = model.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], verbose=0, early_stopping_rounds=25)
    rmse = RMSE(valid_y, lgb_model.predict(valid_X))
    return rmse
       
study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
study_lgb.optimize(objective, n_trials=100)

# val 검증
trial = study_lgb.best_trial
trial_params = trial.params
print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

# test 데이터셋 분할
test_X = test.drop('price', axis=1)
test_y = test['price']

# 데이터에 LightGBM model 적용
final_lgb_model = lgb.LGBMRegressor(**trial_params)
final_lgb_model.fit(train_X, train_y)
final_lgb_pred = final_lgb_model.predict(test_X)

# 평가지표 출력
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print('R2 : {:.3f}'.format(r2_score(test_y, final_lgb_pred)))
print('MSE : {:.3f}'.format(mean_squared_error(test_y, final_lgb_pred)))
print('MAE : {:.3f}'.format(mean_absolute_error(test_y, final_lgb_pred)))
print('RMSE : {:.3f}'.format(np.sqrt(mean_squared_error(test_y, final_lgb_pred))))

score = rmse_cv(final_lgb_model)

# plt 한글 폰트 깨짐 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

# 폰트 및 그래프 크기 설정
import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic')
plt.rcParams["figure.figsize"] = [20, 15]

# 그래프 출력
plt.barh(train_X.columns, final_lgb_model.feature_importances_)

# 학습시킨 모델 저장하기
import joblib

joblib.dump(final_lgb_model, './final_model.pkl')

# 피클파일 다시 불러오기
modelReload = joblib.load('final_model.pkl')

# 예측 값 출력 예비 테스트
def make_bus_dict(dong):
    '''버스 노선 사전에서 동(key)를 넣으면 노선수(value)를 리턴하는 함수'''
    # 버스 노선 파일 불러오기
    bus_line_data = pd.read_excel("../data/bus_line_data.xlsx")

    # 각 컬럼별 값들 리스트로 만들기
    adr_dong = list(bus_line_data['adr_dong'])
    bus_num = list(bus_line_data['버스노선수'])

    # dict로 키-값 형태로 저장하기
    bus_dict = {}

    for i in range(0, len(adr_dong)):
        bus_dict[adr_dong[i]] = bus_num[i]

    return bus_dict[dong]


data_dong = '고양동'
data_year = 2023
data_area = 85

# 버스노선수 dict에서 뽑기
data_bus = make_bus_dict(data_dong)

temp = {
    'year' : data_year,
    '전용면적' : data_area,
    '버스노선수' : data_bus,
    '가좌동' : 0,
    '고양동' : 0,
    '관산동' : 0,
    '대화동' : 0,
    '덕은동' : 0,
    '덕이동' : 0,
    '도내동' : 0,
    '동산동' : 0,
    '마두동' : 0,
    '백석동' : 0,
    '사리현동' : 0,
    '삼송동' : 0,
    '성사동' : 0,
    '성석동' : 0,
    '식사동' : 0,
    '신원동' : 0,
    '원흥동' : 0,
    '일산동' : 0,
    '장항동' : 0,
    '주교동' : 0,
    '주엽동' : 0,
    '중산동' : 0,
    '지축동' : 0,
    '탄현동' : 0,
    '토당동' : 0,
    '풍동' : 0,
    '행신동' : 0,
    '향동동' : 0,
    '화정동' : 0
}

# 해당 동은 1로 바꿔주기
temp[data_dong] = 1

# dataframe(series) 형태로 변환
data_f = pd.DataFrame([temp])

# model에 넣어서 예측값 구하기
modelReload.predict(data_f)



