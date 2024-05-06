# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Import the required packages and print the present data
3. Print the placement data and salary data.
4. Find the null and duplicate values.
5. Using logistic regression find the predicted values of accuracy , confusion matrices.
6. Stop

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VEMBARASAN P
RegisterNumber: 212223220123
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/rohithprem/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
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
![1](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/38233e24-9e44-4a73-bbd4-e6f2eeff17cd)
![2](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/ff502714-439d-42a7-82cb-ca172496973a)
![3](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/fc5d9131-8f99-4e48-9373-c212025c6c9e)
![4](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/8b1a45c2-5ffa-45f0-8b20-ba3ad2dfd08f)
![5](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/baab7f26-60b1-467a-b783-5acb3994427f)
![6](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/4ef74eef-f93e-4b8c-ac7c-c68bc731b279)
![7](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/1f74d6d5-dfae-4e03-8991-484d014f32f1)
![8](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/08339dfc-316f-41f8-a9f6-bf2ae2ae3a2f)
![9](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/5770d9ac-6e44-452f-8717-6c3780efa0fa)
![10](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/54940b22-1e90-4eef-bb8c-eb9a4ae82519)
![11](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/58a5fdf3-fb3c-43a6-8700-35ab7079cb50)
![12](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/146315115/bc894fe8-84b3-431d-83ae-0263d9fddff3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
