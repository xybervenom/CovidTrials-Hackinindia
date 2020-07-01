import pandas as pd
# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
df=pd.read_excel("covid.xlsx")
df.isnull().sum().max()
from sklearn import svm
clf= svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)