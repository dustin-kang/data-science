#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 전체 컬럼 출력하기
pd.set_option('display.max_columns', None)

df = pd.read_csv('../data/players_21.csv')
cls4_df = pd.read_csv('../data/cls_data_4p.csv')
cls11_df = pd.read_csv('../data/cls_data_11p.csv')
reg_df = pd.read_csv('../data/reg_data.csv')


# In[25]:


cls11_df


# In[15]:


get_ipython().system('pip install plotly')


# In[16]:


import plotly.express as px


# In[17]:


# for col in df.columns:
#   fig = px.box(df, y=f'{col}', title= f'{col}')
#   fig.show()


# In[18]:


px.scatter(df, x="international_reputation", y = "value_eur", title = 'Players Value for International_reputation (인기도에 따른 선수들 가치)')


# In[19]:


px.scatter(df, x="overall", y = "value_eur", title = 'Players Value for overall score(전반 점수에 따른 선수들 가치)' )


# In[21]:


px.scatter(df, x="release_clause_eur", y = "value_eur", title = 'Players Value for Release Clause(릴리즈 조항에 따른 선수들 가치)' )


# In[ ]:




