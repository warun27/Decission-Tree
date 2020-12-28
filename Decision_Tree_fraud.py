# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 02:58:11 2020

@author: shara
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
fraud = pd.read_csv("F:\Warun\DS Assignments\DS Assignments\Decision Tree\Fraud_check.csv")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn import preprocessing
fraud.info()
output= []
a = "Risky"
b = "Good"
for i in fraud["Taxable.Income"]:
    if i <= 30000:
        output.append(a)
    else:
        output.append(b)
        
fraud["Output"] = output
fraud["Output"].value_counts()
print(fraud.head())
categorical_column = ["Marital.Status", "Urban","Undergrad"]
fraud.isna().sum()

fraud.head()
fraud_dummy = pd.get_dummies(fraud, columns = categorical_column)
fraud_dummy.Output.value_counts()
sns.countplot(fraud_dummy.Output)

fraud.Undergrad.value_counts()
sns.countplot(fraud.Undergrad)

fraud['Marital.Status'].value_counts()
sns.countplot(fraud['Marital.Status'])

fraud['Taxable.Income'].value_counts()
sns.countplot(fraud['Taxable.Income'])

fraud.Urban.value_counts()
sns.countplot(fraud.Urban)

f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(fraud_dummy.corr(), annot=True, fmt='.2f')

sns.boxplot(data = fraud, orient = "n", palette = "Set3")
fraud.boxplot(return_type = 'axes', figsize = (30,10))

column_list = []
iqr_list = []
out_low = []
out_up = []
tot_outlier = []


for i in fraud_dummy.describe().columns : 
    QTR1 = fraud_dummy[i].quantile(0.25)
    QTR3 = fraud_dummy[i].quantile(0.75)
    IQR = QTR3 - QTR1
    LTV = QTR1 - (1.5* IQR)
    UTV = QTR3 + (1.5 * IQR)
    current_column = i
    current_iqr = IQR
    bl_LTV = fraud_dummy[fraud_dummy[i] < LTV][i].count()
    ab_UTV = fraud_dummy[fraud_dummy[i] > UTV][i].count()
    TOT_outliers = bl_LTV + ab_UTV
    column_list.append(current_column)
    iqr_list.append(current_iqr)
    out_low.append(bl_LTV)
    out_up.append(ab_UTV)
    tot_outlier.append(TOT_outliers)
    outlier_report = {"Column_name" : column_list, "IQR" : iqr_list, "Below_outliers" : out_low, "Above_outlier" : out_up, "Total_outliers" : tot_outlier}
    outlier_report = pd.DataFrame(outlier_report)
    print(outlier_report)
    
sns.boxplot(data = fraud_dummy['Taxable.Income'] , orient = "n", palette = "Set3")
sns.boxplot(data = fraud_dummy['City.Population'] , orient = "n", palette = "Set3")
sns.boxplot(data = fraud_dummy['Work.Experience'], orient = "n", palette = "Set3")

bins = range(0,100,10)

sns.distplot(fraud_dummy['Taxable.Income'][fraud.Output == "Good"] , bins = bins)
sns.distplot(fraud_dummy['Taxable.Income'][fraud.Output == "Risky"] , bins = bins)
sns.distplot(fraud_dummy['City.Population'][fraud.Output == "Good"] , bins = bins)
sns.distplot(fraud_dummy['City.Population'][fraud.Output == "Risky"] , bins = bins)
sns.distplot(fraud_dummy['Work.Experience'][fraud.Output == "Good"] , bins = bins)
sns.distplot(fraud_dummy['Work.Experience'][fraud.Output == "Risky"] , bins = bins)


x = fraud_dummy.drop("Output", axis = 1)
y = fraud_dummy["Output"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20,random_state = 0)
dtree = DecisionTreeClassifier(criterion='entropy', random_state=1)
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 7)
cross_val_train = model_selection.cross_val_score(dtree, x_train,y_train, cv = kfold, scoring = "accuracy")
cross_val_test = model_selection.cross_val_score(dtree, x_test,y_test, cv = kfold, scoring = "accuracy")
cls_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=(True)))
np.mean(y_pred ==y_test)
