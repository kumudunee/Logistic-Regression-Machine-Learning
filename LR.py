import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insuarance.csv")
print(df)

plt.scatter(df.age,df.Bought_insuarance)
plt.show()

print(df.shape)

x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.Bought_insuarance, test_size=0.1)
print(x_test)

print(x_train)

model = LogisticRegression()
print(model.fit(x_train,y_train))

print(model.predict(x_test))

print(model.score(x_test,y_test))

print(model.predict_proba(x_test))

print(model.predict([[55]]))

print(model.predict([[2]]))
