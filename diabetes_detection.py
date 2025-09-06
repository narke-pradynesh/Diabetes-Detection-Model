# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

df.info()

X = df.drop('Class', axis=1).values
y = df['Class'].values

"""Splitting dataset"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""###Feature Scaling"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc.transform(X_test[:, 1:])

"""###Model creation"""

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
random_forest_model = rf_classifier.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)

"""###Evaluation"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print(f'Random forest confusion matrix:\n{cm}')

ConfusionMatrixDisplay(cm).plot(cmap='flare')
plt.show()

cr = classification_report(y_test, y_pred)
print(f'Random Forest Classification Report:\n{cr}')

print(f'Random forest accuracy score:\n{accuracy_score(y_test, y_pred)}')

rf_classifier.feature_importances_

"""###Optimization using important features"""

imp_features = ['AGE','HbA1c', 'BMI']
X = df[imp_features].values
y = df['Class'].values

"""###Exporting model"""

import joblib
joblib.dump(rf_classifier, 'model.pkl')

"""#Checking model"""

import seaborn as sns
print(df.corr()["Class"].sort_values(ascending=False))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)
print("Cross-val scores:", scores)
print("Mean Accuracy:", scores.mean())