#!/usr/bin/env python
# coding: utf-8

# # import Module

# In[170]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/players_21.csv')


# In[171]:


df.shape


# # EDA

# In[172]:


df.describe()


# **축구선수들의 평균 신체적 능력**
# - 나이 : 25세
# - 키 : 181.1cm
# - 몸무게 : 75kg
# - 평균 임금 : 8,675유로 (약 천만원)
# 

# ## Cleaning 
# ```
# `sofifa_id`, `player_url`, `long_name`, `dob`, 'player_tags', `club_name`, `league_name`, `loaned_form`, `joined`, `contract_valid_until`, `nation_jersey_number`,`player_traits`, `defending_marking`, `nationality`
# ```
# 
# 상위 특성들(features)은 다중공선성이 있거나 분석 및 예측에 불필요한 특징이라 판단하여 제거함.
# 
# - 긴이름(long_name)과 짧은 이름(short_name)이 겹치므로 제거함.
# - 생일(dob)과 나이(age)가 특성이 겹치므로 제거함.
# - 선수들의 국가팀(nationality)과 리그팀(leage)이 따로 존재하므로 분석에는 불필요하므로 제거함.
# - player_tag는 다른 컬럼을 통해 수치적으로 알 수 있는 컬럼이고, 특수한 선수들에게만 '#tag'가 주어지기 때문에 제거함.
# 
# 그외로, 선수들이 팀에 들어간 년도, 등번호 등이 있음.

# In[173]:


del_features = ['player_url', 'long_name', 'dob', 'player_tags', 'club_name', 'league_name', 'loaned_from', 'joined', 'contract_valid_until', 'nation_jersey_number','player_traits','nationality', 'defending_marking','team_jersey_number','league_rank']
df = df.drop(del_features, axis=1)


# In[174]:


df.head()


# ## Integration
# 
# 분류(Classifier)과 회귀(Regression)에 따라 데이터를 어떻게 처리해야할 지 나눠지게 됨.
# 
# ### Classifier
# - team_position : 포지션들이 단일화 되어있고, 결측치는 255개임.
# - nation_position : 결측치가 17,817개가 있음. 보통 선수들은 리그전에서 뽑힌 선수들을 국가전에 사용되는 것으로 알고 있음.
# - player_position : 결측치는 없으나 중복되는 포지션이 있음.
# 
# > 우선, `Team_position`으로 데이터를 합치고 결측치는 `player_poition`에서 우선도가 높은 포지션을 채우기로 함.
# 
# ### Regression
# - value_eur : 선수들의 몸값
# - wage_eur : 선수들의 임금 값
# - release_clause_eur : 릴리즈 조항, 어떤 팀에서 선수가 강등을 당하거나 다른 선수를 필요로 할 때, 계약기간 없이 데리고 올 수 있는 최소 금액. (보통, NameValue가 있는 선수들에게만 해당.)

# ## Select Target data

# ### Classifer

# In[175]:


df_player_position = pd.DataFrame(df['player_positions'].str.split(',',3).tolist()) # 'player_position'의 ','를 제거
df['player_positions'] = df_player_position[0] # 가장 우선이 되는 포지션으로 변경


# - `,` 구분자를 이용해 컬럼을 나눠 선정하지 않은 2개의 컬럼은 결측치가 80%가 넘으므로 제외함.
# - `nation_position`은 다른 컬럼 보다 [SUB](https://en.wikipedia.org/wiki/Substitute_(association_football))선수가 많으므로 제외함.

# In[176]:


team_position_nan = df[df['team_position'].isna()] # 리그팀 포지션 결측치


# In[177]:


df['team_position'][df['team_position'].isna()] = team_position_nan.loc[:,'player_positions'] # 포지션 채우기


# In[178]:


df['team_position'].isna().sum()


# In[179]:


df = df.drop(columns=['player_positions','nation_position']) # 다른 포지션 컬럼 제거


# In[180]:


df.rename(columns={"team_position": "position"},inplace=True) # 이름 변경


# ### Regression

# In[181]:


df = df.drop(columns = 'release_clause_eur', axis=1) # 릴리즈 조항 제거
df


# # Profiling

# 판다스 프로파일링을 통해 간단한 EDA View를 확인해봄.
# ```
# !pip install pandas-profiling==2.7.1
# ```

# In[218]:


data = df.copy()


# In[220]:


from pandas_profiling import ProfileReport

df_report = ProfileReport(df)
df_report.to_file('./profile_report.html')


# # Feature Engineering

# In[221]:


data.head()


# ### work_rate
# - 설명 : 공격 / 수비 비율이 한 특성으로 이루어져 있음. e.g._Medium/Low, High/Low_
# - 특징 : 중간 / 중간이라는 비율의 선수들이 많이 높음.
# 
# > 공격 비율과 수비 비율 특성, 두가지 특성으로 나눔.

# In[222]:


data['work_rate'].value_counts(normalize=True)


# In[223]:


data['work_rate'].isna().sum() # 결측치는 없음.


# In[224]:


work_rate = pd.DataFrame(data['work_rate'].str.split('/',2).tolist()).rename({0:'work_rate_attaking', 1: 'work_rate_defensive'},axis=1)
# 공격과 수비 두 비율의 데이터 프레임으로 바꿈

data['work_rate_Attaking'] = work_rate['work_rate_attaking'] # 공격 비율
data['work_rate_defensive'] = work_rate['work_rate_defensive'] # 수비 비율


# ### body_type
# - 설명 : 선수들의 체형은 Normal  / Lean / Stokcy / PLAYER_BODY_TYPE_n / 유니크한 선수들 이름 으로 이루어져있음.
# 
# > - PLAYER_BODY_TYPE_N으로 되어있는 선수들은 Normal 체형으로 통일화 함. 
# > - 선수들의 이름으로 이루어진 체형들은 Unique라는 특성으로 범주를 통일시킴.

# In[225]:


data['body_type'].value_counts()


# In[226]:


data['body_type'] = data['body_type'].str[:16]  # 'PLAYER_BODY_TYPE_266'에서 266만 제거하는 문자열 수정
data['body_type'][(data['body_type'] != 'Normal') &
                  (data['body_type'] != 'Stocky') &
                  (data['body_type'] != 'Lean') &
                  (data['body_type'] != 'PLAYER_BODY_TYPE')] = 'Unique' # 이외 해당하지 않는 특수한 선수 데이터는 'Unique'한 데이터로 설정


# In[227]:


data['body_type'][(data['body_type'] != 'Normal') &
                  (data['body_type'] != 'Stocky') &
                  (data['body_type'] != 'Lean') &
                  (data['body_type'] != 'PLAYER_BODY_TYPE')] = 'Unique'


# In[228]:


data['body_type'] = data['body_type'].replace('PLAYER_BODY_TYPE','Normal') # 'PLAYER_BODY_TYPE'인 선수들은 Normal로 바꾼다.
data['body_type'].value_counts()


# ### ls ~ rb
# - 설명 : ls 부터 rb까지는 선수들의 기존 능력치에 +n 만큼 잠재력을 추가한 것임.
# - for 문을 통해 아래와 같이 성장값(grow)과 기존 값(value)으로 열을 나누기로 함.
# > 21+3 => value = 21, grow = 3

# In[229]:


data_pos_score = data.iloc[:, 62:88] # '능력치 + n' 특성


# In[231]:


for n in range(len(data_pos_score.columns)):
    # '+' 구분자를 중심으로 기존 값과 성장값으로 나눔 
  data[f'{data_pos_score.columns[n].upper()}_value'] = pd.DataFrame(data_pos_score.iloc[:,n].str.split('+',2).tolist())[0].astype(int)
  data[f'{data_pos_score.columns[n].upper()}_grow'] = pd.DataFrame(data_pos_score.iloc[:,n].str.split('+',2).tolist())[1]
  # ''인 데이터들은 0으로 값을 변경함.  
  data[f'{data_pos_score.columns[n].upper()}_grow'][data[f'{data_pos_score.columns[n].upper()}_grow'] == ''] = 0
  data[f'{data_pos_score.columns[n].upper()}_grow'] = data[f'{data_pos_score.columns[n].upper()}_grow'].astype(int)


# In[232]:


data.drop(columns = data.iloc[:,62:88], axis=1,inplace=True) # 기존에 있었던 특성들은 제거 함.


# In[233]:


data.iloc[:,64:123]


# In[236]:


data.describe()


# ### goalkeeper Data
# - `pace`,	`shooting`,	`passing`	,`dribbling`	,`defending`	,`physic` 열에 있는 결측치는 골기퍼들의 데이터임.
# 
# - 골기퍼 관련 데이터들은 반대로 골기퍼 포지션 선수들의 데이터 밖에 없음.

# In[237]:


pd.DataFrame(data[['gk_diving','goalkeeping_diving']])


# In[240]:


data.drop(data.iloc[:,22:28], inplace=True, axis=1)


# In[248]:


data


# In[249]:


data.to_csv('../data/processing_data.csv')

