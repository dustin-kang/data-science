#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/dustin-kang/Proj2_SoccerPlayer-Machine-Learning/blob/main/Report/6_Hyperparameter_Tuning_Classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Hyperparameter_Tuning_Classifier
# 
# 파라미터 최적화를 통해 더 좋은 성능을 내기 위해 모델을 조정한다.
# 
# 
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

df4 = pd.read_csv('../data/cls_data_4p.csv')
df11 = pd.read_csv('../data/cls_data_11p.csv')


# ## 1. Train, Test, Split

# In[12]:


from sklearn.model_selection import train_test_split 

target = 'position'
train, test = train_test_split(df11, random_state=2, train_size=.75, stratify=df11[target])
train, val = train_test_split(train, random_state=2, train_size=.75, stratify=train[target])
train.shape, val.shape, test.shape


# In[13]:


features = train.columns.drop(target)
# X (features)
X_train = train[features] 
X_val = val[features]
X_test = test[features]

# y (Target)
y_train = train[target]
y_val = val[target] 
y_test = test[target]


# ## pre-OneHot Encoding

# In[14]:


# preprocessing

from sklearn.pipeline import make_pipeline, Pipeline
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Classifier Model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, f1_score, accuracy_score

# Model Selection, Parameter Tuning
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[15]:


enc = OneHotEncoder(cols = [
                            'preferred_foot',
                            'body_type'
], use_cat_names=True)

enc_train = enc.fit_transform(X_train)
train_features_transform = enc_train.columns # 인코딩이 완료된 특성을 변수에 담기

enc_val = enc.transform(X_val)
enc_test = enc.transform(X_test)


# ## RandomizedSearchCV
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=qbxlvnf11&logNo=221574025920

# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(50, 500),
  'randomforestclassifier__max_depth' : [5, 10, 15, 20, None],
  'randomforestclassifier__min_samples_leaf' : [5, 10, 15, 20, None],
  'randomforestclassifier__min_samples_split' : [5, 15, 30, 40, None]
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=50, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


clf.best_params_


# In[ ]:


clf.best_score_


# In[ ]:


pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(50, 500),
  'randomforestclassifier__max_depth' : randint(10,25),
  'randomforestclassifier__min_samples_leaf' : randint(5,10),
  'randomforestclassifier__min_samples_split' : [5, 10, 15]
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=50, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


clf.best_params_


# In[ ]:


clf.best_score_


# In[ ]:


pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(370, 430),
  'randomforestclassifier__max_depth' : randint(14,24),
  'randomforestclassifier__min_samples_leaf' : randint(5,10),
  'randomforestclassifier__min_samples_split' : randint(5, 10)
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=50, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


clf.best_score_


# In[ ]:


clf.best_estimator_


# In[ ]:


model = clf.best_estimator_


# In[ ]:


pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(380, 410),
  'randomforestclassifier__max_depth' : randint(18,24),
  'randomforestclassifier__min_samples_leaf' : [5,6],
  'randomforestclassifier__min_samples_split' : randint(5, 10)
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=10, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))


# In[ ]:


pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(100, 300),
  'randomforestclassifier__max_depth' : randint(10,21),
  'randomforestclassifier__min_samples_leaf' : [5,6],
  'randomforestclassifier__min_samples_split' : randint(5, 10),
  'randomforestclassifier__max_features' : randint(3,5)
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=10, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))


# In[ ]:


pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(80, 200),
  'randomforestclassifier__max_depth' : randint(5,15),
  'randomforestclassifier__min_samples_leaf' : [5,6],
  'randomforestclassifier__min_samples_split' : randint(5, 10),
  'randomforestclassifier__max_features' : randint(3,6)
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=10, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))


# In[ ]:


pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'randomforestclassifier__n_estimators' : randint(100, 200),
  'randomforestclassifier__max_depth' : randint(10,14),
  'randomforestclassifier__min_samples_leaf' : randint(5,10),
  'randomforestclassifier__min_samples_split' : randint(7, 10),
  'randomforestclassifier__max_features' : randint(3,8)
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    RandomForestClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=12, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))

pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


# Modeling

dists = { # Randomized Classifier Parameter
  'xgbclassifier__n_estimators' : randint(130,180),
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=3, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))

pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


dists = { # Randomized Classifier Parameter
  'xgbclassifier__n_estimators' : randint(150,180),
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=3, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('Random Forest Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))

pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


dists = { # Randomized Classifier Parameter
  'xgbclassifier__n_estimators' : randint(160,178),
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2)
)

clf = RandomizedSearchCV( # RamdommizedSearchCV
    pipe, 
    param_distributions=dists, 
    n_iter=3, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('XGBoost Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))

pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


param = { # Randomized Classifier Parameter
  'xgbclassifier__n_estimators' : (166,169,171,174,175,178),
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2)
)

clf = GridSearchCV( # RamdommizedSearchCV
    pipe, 
    param_grid=param, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('XGBoost Classifier (Position)', fontsize=15);


# In[ ]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))

pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[ ]:


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2, n_estmators=166)
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))


# # Permutation importances
# <aside>
# 
# ###💡 **5) 머신러닝 모델 해석**
# 
# 프로젝트에서 가장 중요하다고 볼 수 있는 부분 입니다. 우리는 SHAP, PDP 등을 통해서 모델이 관측치를 어떤 특성을 활용했거나, 어떤 특성이 타겟에 영향을 끼쳤는지 등을 해석하는 방법에 대해서 배웠습니다.
# 여러분의 프로젝트에도 이러한 해석 방법을 활용해 머신러닝 모델을 비전문가라도 조금 더 쉽게 이해하고 접근할 수 있도록 해주셔야 합니다.
# 
# - PDP, SHAP을 활용하여 최종 모델을 설명합니다
# - 시각화는 "설명"이 제일 중요합니다.
# 
# ### **태스크를 수행한 후, 다음 질문에 대답할 수 있어야 합니다.**
# 
# 1. 모델이 관측치를 예측하기 위해서 어떤 특성을 활용했나요?
# 2. 어떤 특성이 있다면 모델의 예측에 도움이 될까요? 해당 특성은 어떻게 구할 수 있을까요?
# 
# train_features_transform
# </aside>

# In[12]:


get_ipython().system('pip install eli5')


# In[13]:


pipe = Pipeline([
                 ('preprocessing', make_pipeline(
                     OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
                     SimpleImputer(missing_values=np.nan, strategy='mean')
                 )),
                 ('clf',  XGBClassifier(random_state=2, n_estmators=166))
],verbose=1)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


# In[14]:


import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(
    pipe.named_steps['clf'],
    scoring='accuracy',
    n_iter=5,
    random_state=2
)

X_test_transformed = pipe.named_steps['preprocessing'].transform(X_test)
permuter.fit(X_test_transformed, y_test);

feature_names = train_features_transform.tolist() #Xval의 컬럼 *리스트화*
pd.Series(permuter.feature_importances_, feature_names).sort_values()


# In[15]:


# 특성별 score 확인
eli5.show_weights(
    permuter, 
    top=None, # top n 지정 가능, None 일 경우 모든 특성 
    feature_names=feature_names # list 형식으로 넣어야 합니다
)


# In[17]:


print('특성 삭제 전:', X_train.shape, X_val.shape)

minimum_importance = 0.000 # 최소 중요도 (최소 이정도 이상은 넘어야 한다.)
mask = permuter.feature_importances_ > minimum_importance
features = train_features_transform[mask]

X_train_selected = enc_train[features]
X_val_selected = enc_val[features]
X_test_selected = enc_test[features]

print('특성 삭제 후:', X_train_selected.shape, X_val_selected.shape)


# In[20]:


pipe = Pipeline([
                 ('preprocessing', make_pipeline(
#                     OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
                     SimpleImputer(missing_values=np.nan, strategy='mean')
                 )),
                 ('clf',  XGBClassifier(random_state=2, n_estmators=166))
],verbose=1)

pipe.fit(X_train_selected, y_train)
y_pred = pipe.predict(X_val_selected)

y_train_pred = pipe.predict(X_train_selected)

print(classification_report(y_val, y_pred))

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_pred, average='macro'))


# In[22]:


y_pred = pipe.predict(X_test_selected)

y_train_pred = pipe.predict(X_train_selected)

print(classification_report(y_test, y_pred))

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('테스트 f1 score (micro): ', f1_score(y_test, y_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('테스트 f1 score (macro): ', f1_score(y_test, y_pred, average='macro'))
print('---------------')
print('훈련 accuracy (micro): ', accuracy_score(y_train, y_train_pred))
print('테스트 accuracy (micro): ', accuracy_score(y_test, y_pred))

print('훈련 accuracy (macro): ', accuracy_score(y_train, y_train_pred))
print('테스트 accuracy (macro): ', accuracy_score(y_test, y_pred))

# 시각화
label = ['Attack_C', 'Defence_C', 'Midfielder_C', 'Def_Midfielder', 'GK',
         'Attack_L', 'Defence_L', 'Midfielder_L', 'Attack_R', 'Defence_R', 'Midfielder_R']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(pipe, 
                              X_test_selected, y_test,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)

plt.title('XGBoost Classifier (Position)', fontsize=15);


# In[23]:


get_ipython().system('pip install shap')


# In[47]:


enc = OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

clf = XGBClassifier(random_state=2, n_estmators=166)

X_train_transformed = enc.fit_transform(X_train)
X_test_transformed = enc.transform(X_test)
X_train_imputed = imp.fit_transform(X_train_transformed)
X_test_imputed = imp.transform(X_test_transformed)

clf.fit(X_train_imputed, y_train)
y_pred = clf.predict(X_test_imputed)


# In[43]:


dataset = pd.DataFrame(X_test_transformed)


# In[44]:


dataset.columns = enc_test.columns


# In[53]:


import shap

explainer = shap.TreeExplainer(clf) 
shap.initjs() #인라인 자바스크립트

shap_values = explainer.shap_values(dataset.iloc[:200])
shap.summary_plot(shap_values, dataset.iloc[:200], plot_type='bar', max_display=6)


# In[37]:


X_train_selected


# # 4 Positions

# In[69]:


df = pd.read_csv('/content/cls_df.csv')
df = df.iloc[:,1:]


# In[70]:


df


# In[71]:


# 공격수
df.loc[df['position']=='ST', ['position']] = 'Attack'
df.loc[df['position']=='LS', ['position']] = 'Attack'
df.loc[df['position']=='RS', ['position']] = 'Attack'
df.loc[df['position']=='RF', ['position']] = 'Attack'
df.loc[df['position']=='LF', ['position']] = 'Attack'
df.loc[df['position']=='CF', ['position']] = 'Attack'
df.loc[df['position']=='LW', ['position']] = 'Attack'
df.loc[df['position']=='RW', ['position']] = 'Attack'

# 골기퍼
df.loc[df['position']=='GK', ['position']] = 'Goalkeeper'

# 미드필더
df.loc[df['position']=='CAM', ['position']] = 'Middle'
df.loc[df['position']=='LAM', ['position']] = 'Middle'
df.loc[df['position']=='RAM', ['position']] = 'Middle'
df.loc[df['position']=='AM', ['position']] = 'Middle'
df.loc[df['position']=='LM', ['position']] = 'Middle'
df.loc[df['position']=='RM', ['position']] = 'Middle'
df.loc[df['position']=='CM', ['position']] = 'Middle'
df.loc[df['position']=='LCM', ['position']] = 'Middle'
df.loc[df['position']=='RCM', ['position']] = 'Middle'
df.loc[df['position']=='CDM', ['position']] = 'Middle'
df.loc[df['position']=='LDM', ['position']] = 'Middle'
df.loc[df['position']=='RDM', ['position']] = 'Middle'

# 수비수
df.loc[df['position']=='LWB', ['position']] = 'Defence'
df.loc[df['position']=='RWB', ['position']] = 'Defence'
df.loc[df['position']=='CB', ['position']] = 'Defence'
df.loc[df['position']=='LCB', ['position']] = 'Defence'
df.loc[df['position']=='RCB', ['position']] = 'Defence'
df.loc[df['position']=='LB', ['position']] = 'Defence'
df.loc[df['position']=='RB', ['position']] = 'Defence'


# In[72]:


df.to_csv('/content/cls4_df.csv', index=True)


# In[75]:


from sklearn.model_selection import train_test_split 

target = 'position'
train, test = train_test_split(df, random_state=2, train_size=.75, stratify=df[target])
train, val = train_test_split(train, random_state=2, train_size=.75, stratify=train[target])
train.shape, val.shape, test.shape


# In[76]:


features = train.columns.drop(target)
X_train = train[features] 
X_val = val[features]
X_test = test[features]
y_train = train[target]
y_val = val[target] 
y_test = test[target]


# In[77]:


enc = OneHotEncoder(cols = [
                            'preferred_foot',
                            'body_type'
], use_cat_names=True)

enc_train = enc.fit_transform(X_train)
train_features_transform = enc_train.columns # 인코딩이 완료된 특성을 변수에 담기

enc_val = enc.transform(X_val)
enc_test = enc.transform(X_test)


# In[84]:


param = { # Randomized Classifier Parameter
  'xgbclassifier__n_estimators' : (166,169,171,174,175,178),
}


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2)
)

clf = GridSearchCV( # RamdommizedSearchCV
    pipe, 
    param_grid=param, 
    cv=3, 
    scoring='accuracy',  
    verbose=1,
    n_jobs=-1
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

print(classification_report(y_val, y_pred))


# In[85]:


# 시각화
label = ['Attack', 'Defence', 'Goalkeeper','Middle']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('XGBoost Classifier (Position)', fontsize=15);


# In[86]:


model = clf.best_estimator_

y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_val_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_val_pred, average='macro'))

pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score').T


# In[88]:


pipe = make_pipeline( # Modeling Pipeline
    OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
    SimpleImputer(missing_values=np.nan, strategy='mean'),
    XGBClassifier(random_state=2, n_estmators=178)
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))


# In[91]:


y_pred = pipe.predict(X_test)

y_train_pred = pipe.predict(X_train)

print(classification_report(y_test, y_pred))

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('테스트 f1 score (micro): ', f1_score(y_test, y_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('테스트 f1 score (macro): ', f1_score(y_test, y_pred, average='macro'))
print('---------------')
print('훈련 accuracy (micro): ', accuracy_score(y_train, y_train_pred))
print('테스트 accuracy (micro): ', accuracy_score(y_test, y_pred))

print('훈련 accuracy (macro): ', accuracy_score(y_train, y_train_pred))
print('테스트 accuracy (macro): ', accuracy_score(y_test, y_pred))


# 시각화
label = ['Attack', 'Defence', 'Goalkeeper','Middle']
fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(clf, 
                              X_val, y_val,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)
plt.title('XGBoost Classifier (Position)', fontsize=15);


# In[93]:


pipe = Pipeline([
                 ('preprocessing', make_pipeline(
                     OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
                     SimpleImputer(missing_values=np.nan, strategy='mean')
                 )),
                 ('clf',  XGBClassifier(random_state=2, n_estmators=178))
],verbose=1)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


# In[94]:


import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(
    pipe.named_steps['clf'],
    scoring='accuracy',
    n_iter=5,
    random_state=2
)

X_test_transformed = pipe.named_steps['preprocessing'].transform(X_test)
permuter.fit(X_test_transformed, y_test);

feature_names = train_features_transform.tolist() #Xval의 컬럼 *리스트화*
pd.Series(permuter.feature_importances_, feature_names).sort_values()


# In[95]:


# 특성별 score 확인
eli5.show_weights(
    permuter, 
    top=None, # top n 지정 가능, None 일 경우 모든 특성 
    feature_names=feature_names # list 형식으로 넣어야 합니다
)


# In[96]:


print('특성 삭제 전:', X_train.shape, X_val.shape)

minimum_importance = 0.000 # 최소 중요도 (최소 이정도 이상은 넘어야 한다.)
mask = permuter.feature_importances_ > minimum_importance
features = train_features_transform[mask]

X_train_selected = enc_train[features]
X_val_selected = enc_val[features]
X_test_selected = enc_test[features]

print('특성 삭제 후:', X_train_selected.shape, X_val_selected.shape)


# In[97]:


pipe = Pipeline([
                 ('preprocessing', make_pipeline(
#                     OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True),
                     SimpleImputer(missing_values=np.nan, strategy='mean')
                 )),
                 ('clf',  XGBClassifier(random_state=2, n_estmators=166))
],verbose=1)

pipe.fit(X_train_selected, y_train)
y_pred = pipe.predict(X_val_selected)

y_train_pred = pipe.predict(X_train_selected)

print(classification_report(y_val, y_pred))

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('검증 f1 score (micro): ', f1_score(y_val, y_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('검증 f1 score (macro): ', f1_score(y_val, y_pred, average='macro'))


# In[99]:


y_pred = pipe.predict(X_test_selected)

y_train_pred = pipe.predict(X_train_selected)

print(classification_report(y_test, y_pred))

print('훈련 f1 score (micro): ', f1_score(y_train, y_train_pred, average='micro'))
print('테스트 f1 score (micro): ', f1_score(y_test, y_pred, average='micro'))

print('훈련 f1 score (macro): ', f1_score(y_train, y_train_pred, average='macro'))
print('테스트 f1 score (macro): ', f1_score(y_test, y_pred, average='macro'))
print('---------------')
print('훈련 accuracy (micro): ', accuracy_score(y_train, y_train_pred))
print('테스트 accuracy (micro): ', accuracy_score(y_test, y_pred))

print('훈련 accuracy (macro): ', accuracy_score(y_train, y_train_pred))
print('테스트 accuracy (macro): ', accuracy_score(y_test, y_pred))

# 시각화
label = ['Attack', 'Defence', 'Goalkeeper','Middle']
fig, ax = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(pipe, 
                              X_test_selected, y_test,
                              display_labels = label,
                              cmap = 'Blues',
                              normalize='true',
                              ax = ax)

plt.title('XGBoost Classifier (Position)', fontsize=15);


# In[100]:


enc = OneHotEncoder(cols=['preferred_foot','body_type'], use_cat_names=True)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

clf = XGBClassifier(random_state=2, n_estmators=166)

X_train_transformed = enc.fit_transform(X_train)
X_test_transformed = enc.transform(X_test)
X_train_imputed = imp.fit_transform(X_train_transformed)
X_test_imputed = imp.transform(X_test_transformed)

clf.fit(X_train_imputed, y_train)
y_pred = clf.predict(X_test_imputed)


# In[101]:


dataset = pd.DataFrame(X_test_transformed)


# In[102]:


dataset.columns = enc_test.columns


# In[104]:


import shap

explainer = shap.TreeExplainer(clf) 
shap.initjs() #인라인 자바스크립트

shap_values = explainer.shap_values(dataset.iloc[:200])
shap.summary_plot(shap_values, dataset.iloc[:200], plot_type='bar', max_display=6)


# In[ ]:




