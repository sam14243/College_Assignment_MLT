import pandas as pd
data = pd.read_csv('Sales.csv')
data = data.dropna()

data = data.replace('Mega', 0)
data = data.replace('Micro', 1)
data = data.replace('Nano', 2)
data = data.replace('Macro', 3)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)

from sklearn.svm import SVR
svr = SVR()
svr.fit(x_train,y_train)

y_pred = svr.predict(x_test)

#print(svr.predict([[78.0, 24.9,4.2,3.0]]))

#%%
import pickle

pickle.dump(svr,open('model.pkl','wb'))
