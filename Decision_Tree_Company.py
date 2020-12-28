# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 02:28:38 2020

@author: shara
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
company = pd.read_csv("G:\DS Assignments\Decision Tree\Company_Data.csv")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics
company.info()
Sales_catg = []

a = "<5"
b = ">=5 and <10"
c = ">10"

for i in company.Sales:
    if i < 5:
        Sales_catg.append(a)
    elif i >= 5 and i < 10:
        Sales_catg.append(b)
    else:
        Sales_catg.append(c)
    
company.Sales_catg = Sales_catg
company["Sales_catg"].value_counts()
categorical_column = ["ShelveLoc", "Urban","US" ]
company_dummy = pd.get_dummies(company, columns = categorical_column)
company_dummy = company_dummy.drop("Sales", axis = 1)
x = company_dummy.drop("Sales_catg", axis = 1)
y = company_dummy["Sales_catg"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20,random_state = 7)


dtree = DecisionTreeClassifier(criterion='entropy', random_state=1)
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)
dtree_acc = np.mean(y_test==y_pred)
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn import preprocessing
kfold = model_selection.KFold(n_splits = 10, random_state = 7)
cross_val = model_selection.cross_val_score(dtree, x_train,y_train, cv = kfold, scoring = "accuracy")
x_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)
dtree.fit(x_train_scaled, y_train)
y_pred_scaled = dtree.predict(x_test_scaled)
cross_val_scaled = model_selection.cross_val_score(dtree, x_train_scaled,y_train, cv = kfold, scoring = "accuracy")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 0)
x_smote, y_smote = smote.fit_resample(x,y)
y_smote.value_counts()
x_train_smote,x_test_smote,y_train_smote,y_test_smote = train_test_split(x_smote,y_smote, test_size = 0.20,random_state = 7)
x_train_smote_scaled = preprocessing.scale(x_train_smote)
x_test_smote_scaled = preprocessing.scale(x_test_smote)
dtree.fit(x_train_smote_scaled, y_train_smote)
cross_val_smote = model_selection.cross_val_score(dtree, x_train_smote_scaled,y_train_smote, cv = kfold, scoring = "accuracy")
y_pred_smote_scaled = dtree.predict(x_test_smote_scaled)
np.mean(y_test_smote == y_pred_smote_scaled)
cross_val_smote_test = model_selection.cross_val_score(dtree, x_test_smote_scaled,y_test_smote, cv = kfold, scoring = "accuracy")
cls_report= pd.DataFrame(classification_report(y_test_smote, y_pred_smote_scaled, output_dict=(True)))
