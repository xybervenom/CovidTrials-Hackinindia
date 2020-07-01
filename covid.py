import pandas as pd
import numpy as np
# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

#Reading and splitting
df=pd.read_excel("covid.xlsx")
X = df.drop(labels='result', axis=1) # Features
y = df.loc[:,'result']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)

#Model
from sklearn import svm
clf= svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)