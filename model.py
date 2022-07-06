import pandas as pd
import numpy as np
import pickle
data=pd.read_csv('HRDataset_v14.csv')
data=data[['Salary','SpecialProjectsCount']]
x=data.loc[:, data.columns != 'Salary']
y=data['Salary']
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(x_train,y_train)
pickle.dump(gbr,open('model.pkl','wb') )
