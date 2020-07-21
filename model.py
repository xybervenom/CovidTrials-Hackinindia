# Importing the libraries
import numpy as np
import pandas as pd
import pickle

#Reading Data
df=pd.read_excel("covid.xlsx")
X = df.drop(labels='result', axis=1) # Features
y = df.loc[:,'result']  

#Splitting Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1,stratify=y)

#Pre Processing
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()

#Fitting model with trainig data
clf.fit(X_train, y_train)

#dump DT model in model.pickle with write-binary mode
pickle.dump(clf, open('model.pkl','wb'))

#create a classifier model 'model' that can be used for prediction
model = pickle.load(open('model.pkl','rb'))

#model can be tested by using predict function as depicted below
#print(model.predict([[2, 9, 6]]))