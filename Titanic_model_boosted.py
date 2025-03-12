# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 18:57:01 2025

@author: gmnmc
"""

from xgboost import XGBClassifier
import pandas as pd

#%%
# Reading in Data
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')
PassengerId = test_data['PassengerId']
#%%
# Data preprocessing
train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

train_data = train_data.join(pd.get_dummies(train_data['Embarked'])).drop('Embarked', axis = 1)
train_data = train_data.map(lambda x: 1 if x is True else 0 if x is False else x)

#%%
y_train = train_data['Survived']
x_train = train_data.drop(['Survived'], axis = 1)
#%%
# Using gradient boosted trees
model = XGBClassifier(
    n_estimators=100,      
    learning_rate=0.01,    
    max_depth=3,          
    objective='binary:logistic',
    use_label_encoder=False
)
model.fit(x_train, y_train)
#%%
# Data preprocessing
test_data['Sex'] = test_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

test_data = test_data.join(pd.get_dummies(test_data['Embarked'])).drop('Embarked', axis = 1)
test_data = test_data.map(lambda x: 1 if x is True else 0 if x is False else x)

#%%
predictions = model.predict(test_data)
print(predictions)

data = {'PassengerId': PassengerId, 'Survived': pd.Series(predictions)}

output = pd.DataFrame(data)
output
#%%
output.to_csv('output.csv', index=False)
# Get a better result than Random Forest method.
# Accuracy of 0.78 vs 0.75 of Random forest
