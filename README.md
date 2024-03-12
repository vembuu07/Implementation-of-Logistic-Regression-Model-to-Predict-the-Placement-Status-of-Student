# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: p.vembarasan
RegisterNumber:  212223220123
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
Placement Data:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/d7c9298e-99cc-43ba-9ed8-4ec49c529fe6)
Salary Data:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/183fe9cf-23bb-4ccc-b23e-b42c13dfff74)
Checking the null() function:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/fc562049-7537-439e-b5f1-48b4cc0c93fc)
Data Duplicate:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/038d4a85-958d-4501-9f9d-e5dde526a696)
Print Data:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/245b1307-3628-4cc0-b374-199dedc3e838)
Data-Status:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/55d92e3a-7616-4908-b10d-d57c313506bb)
Y_prediction array:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/d273b6cc-0ae5-478c-b8f2-1e084a2d2e0f)
Accuracy value:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/319a3a8a-fdbb-437a-b154-f31cc592df58)
Confusion array:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/d34de52c-54bc-400d-ab2c-8a006de1c9b2)
Classification Report:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/f673d913-3872-4a80-af44-18b8daf6da81)
Prediction of LR:
![image](https://github.com/vembuu07/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150772461/5b6d91f4-c95f-4eeb-bbd9-ec1c26b3c43e)

Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
