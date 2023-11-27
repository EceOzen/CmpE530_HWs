# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:12:38 2023

@author: Ece
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
import plotly as py
import plotly.express as px
from plotly.offline import iplot
import plotly.express as px

from scipy.stats import f_oneway

def ssw(data):
    meann = sum(data) / len(data)
    ssw = 0
    for i in range(len(data)):
        ssw += (data[i]-meann)**2
    return meann, ssw



# read data 
data = pd.read_csv("path")
data.head(5)

# data info

data.info()

# data groups for class
pclass_group = data.groupby("Pclass").PassengerId.count()

# create histograms
data.hist(column='Age')
data.hist(column='Pclass')

# create graph for age class correlation
data.plot(x='Pclass', y='Age', style='o')

# find mean of the data 
mean = data.Age.mean()

# fill nan data in age
#data['Age'] = data['Age'].fillna(mean)

#data['Age'] = data['Age'].fillna(50)

data['Age'] = data['Age'].fillna(15)

# create subclasses 
first_class = data [data["Pclass"] == 1]["Age"].to_list()

second_class = data [data["Pclass"] == 2]["Age"].to_list()

third_class = data [data["Pclass"] == 3]["Age"].to_list()

classes = [first_class, second_class, third_class]

# total sum of squares
sst = 0
for clss in range(len(classes)):
    for change in range(len(classes[clss])):
        sst+=((classes[clss][change] - mean)**2)

# degrees of freedom
df_1 = 3 - 1

# variation of groups
ssw_1 = ssw(first_class)
ssw_2 = ssw(second_class)
ssw_3 = ssw(third_class)

# total variation
ssw_full = ssw_1[1] + ssw_2[1] + ssw_3[1]

# degree of freedom
df_2 = 891 - 3

# variation between groups
ssb = 891*((ssw_1[0]-mean)**2+(ssw_2[0]-mean)**2)

# calculating f-score
f_score = (ssb/df_1)/(ssw_full/df_2)

# anova performs
anova = f_oneway(first_class, second_class, third_class)


