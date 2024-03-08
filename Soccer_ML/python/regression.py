#!/usr/bin/env python
# coding: utf-8

# # Model Baseline 
# <aside>
# 
# ### **💡 4) 머신러닝 방식 적용 및 교차검증**
# 
# 데이터의 탐색과 전처리 작업이 끝났다면 **모델링을 통해 베이스라인과의 성능 비교**를 해봅니다.
# 
# - Linear / Tree-based / Ensemble 모델을 학습하세요. (다양하게 시도해보시는 걸 추천합니다.)
# - 평가지표를 계산 후 베이스라인과 비교해보세요.
# - 어느정도 성능이 나왔다면, 교차 검증 (이하 CV)을 통해서 일반화될 가능성이 있는지 확인해봅니다.
# - 모델 성능을 개선하기 위한 다양한 방법을 적용해보세요.
#     - Hyperparameter tuning, etc.
# - 최소 2개 이상의 모델을 만들어서 validation 점수를 보고하세요.
# - 최종 모델의 test 점수를 보고하세요.
# 
# ### **태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다.**
# 
# 1. 모델을 학습한 후에 베이스라인보다 잘 나왔나요? 그렇지 않다면 그 이유는 무엇일까요?
# 2. 모델 성능 개선을 위해 어떤 방법을 적용했나요? 그 방법을 선택한 이유는 무엇인가요?
# 3. 최종 모델에 관해 설명하세요.
# </aside>

# # Spliting Dataset
# - target 값을 `value_eur` 으로 하고 선수들의 능력을 통해 선수들의 가치를 예측할 예정.

# In[11]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# 전체 컬럼 출력하기
pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")

df = pd.read_csv('../data/reg_data.csv')

df.drop(columns=['position'], axis=1, inplace=True)


# In[12]:


df


# In[16]:


from sklearn.model_selection import train_test_split 

# 타겟(target)
target = 'value_eur'


# 훈련 데이터 셋과 테스트 데이터 셋 분리
train, test = train_test_split(df, random_state=2, train_size=.75)
# 훈랸 데이터셋에서 훈련데이터 셋과 검증 데이터 셋 분리
train, val = train_test_split(train, random_state=2, train_size=.75)

# 특성(features)
features = train.columns.drop(target)


# In[17]:


train.shape, val.shape, test.shape


# In[18]:


# X (features)
X_train = train[features] 
X_val = val[features]
X_test = test[features]

# y (Target)
y_train = train[target]
y_val = val[target] 
y_test = test[target]


# ## Regression Modeling

# In[19]:


get_ipython().system('pip install --upgrade category_encoders')


# In[20]:


get_ipython().system('pip install --upgrade xgboost')


# In[21]:


# 파이프라인 #
from sklearn.pipeline import make_pipeline, Pipeline

# 인코더 #
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# 모델링 #
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 평가 #
from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error


# ### make pipeline
# - `str` 데이터 자료형으로 이루어진 컬럼을 0과 1로 바꾸는 **OneHotEncoder(원핫 인코더)**를 사용함.
# - 이후 훈련 데이터셋은 `fit_transform`을 진행함.
# - 검증, 시험 데이터셋은 `transform`을 진행함.
# 

# In[22]:


pipe = make_pipeline(
    OneHotEncoder(cols=['preferred_foot','body_type','work_rate_Attaking','work_rate_defensive'], use_cat_names=True),
    # SimpleImputer(missing_values=np.nan, strategy='mean'),
    MinMaxScaler()
)

X_train_transform = pipe.fit_transform(X_train)
X_val_transform = pipe.transform(X_val)
X_test_transform = pipe.transform(X_test)


# ### Linear Regression (선형회귀)

# In[24]:


reg = LinearRegression()
reg = reg.fit(X_train_transform, y_train)
y_pred = reg.predict(X_val_transform)


# In[25]:


print(f"- MAE : {mean_absolute_error(y_val, y_pred)}")
print(f"- RMSE : {mean_squared_error(y_val, y_pred)**0.5}")
print(f"- R2Score : {r2_score(y_val, y_pred)}")


# In[26]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize=[15,10])
sns.regplot(x=y_val, 
           y=y_pred, 
           fit_reg=True);

plt.title("Linear Regression Plot(Value_EUR)", fontsize=20)
plt.xlabel('Truth', fontsize=14)

plt.ylabel('Prediction', fontsize=14)
plt.show()


# ### Ridge

# In[27]:


reg = Ridge()
reg = reg.fit(X_train_transform, y_train)
y_pred = reg.predict(X_val_transform)

print(f"MAE : {mean_absolute_error(y_val, y_pred)}")
print(f"RMSE : {mean_squared_error(y_val, y_pred)**0.5}")
print(f"R2Score : {r2_score(y_val, y_pred)}")


# In[28]:


fig = plt.figure(figsize=[15,10])
sns.regplot(x=y_val, 
           y=y_pred, 
           fit_reg=True);

plt.title("Ridge Plot(Value_EUR)", fontsize=20)
plt.xlabel('Truth', fontsize=14)

plt.ylabel('Prediction', fontsize=14)
plt.show()


# ### RandomForest (랜덤 포레스트)

# In[29]:


reg = RandomForestRegressor()
reg = reg.fit(X_train_transform, y_train)
y_pred = reg.predict(X_val_transform)

print(f"MAE : {mean_absolute_error(y_val, y_pred)}")
print(f"RMSE : {mean_squared_error(y_val, y_pred)**0.5}")
print(f"R2Score : {r2_score(y_val, y_pred)}")


# ### XGBoost

# In[2]:


reg = XGBRegressor()
reg = reg.fit(X_train_transform, y_train)
y_pred = reg.predict(X_val_transform)

print(f"MAE : {mean_absolute_error(y_val, y_pred)}")
print(f"RMSE : {mean_squared_error(y_val, y_pred)**0.5}")
print(f"R2Score : {r2_score(y_val, y_pred)}")

"""
[23:20:11] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
MAE : 427093.0390904744
RMSE : 1251733.4461512708
R2Score : 0.949373346492528
"""


# ## 결론적으로
# 랜덤포레스트 모델이 가장 적은 오차를 보여주고 있습니다. 
