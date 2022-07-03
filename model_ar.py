#importing the data
import pandas as pd
import pickle
df=pd.read_csv("C:/Users/acer/Downloads/edtech_data.csv")
df.columns
#data preprocessing and EDA
df.describe
df.head
df.shape
import numpy as np
df.dtypes
#dropping the enrolled_students column as it is not necessary for the prediction
df_new=df.drop("enrolled_students",axis=1)
#one hot encoding  and getting dummies for getting the numerical data
df_new["placement"]=pd.get_dummies(df_new.placement,drop_first=True)
 #yes=1  and  no=0  in the placement columns
df_new["course_category"]=pd.get_dummies(df_new.course_category,drop_first=True)
 #pg_course=0  and  skill_enhancement=1  in the course_category columns
df_new["course_type"]=pd.get_dummies(df_new.course_type,drop_first=True)
 #offline=0  and  online=1  in the course_category columns
# one hot encoding for the state and course title
a=pd.get_dummies(df_new.state,drop_first=True)
df_new=pd.concat([df_new,a],axis=1)
#drop state column
df_new.drop("state",axis=1,inplace=True)
#now for the course title
b=pd.get_dummies(df_new.course_title,drop_first=True)
df_new=pd.concat([df_new,b],axis=1)
#drop course_title column
df_new.drop("course_title",axis=1,inplace=True)
#EDA (exploratory data analysis)
import matplotlib.pyplot as plt
import seaborn as sns
#correlatin matrix
corr_matrix=df_new.corr()
print(corr_matrix["price"].sort_values(ascending=False))
#dropping the electricity_and_other_charges as it not corelated
df_new.drop("electricity_and_other_charges",axis=1,inplace=True)

#constructing a heat map for finding the correlation
plt.figure(figsize=(8,8))
sns.heatmap(corr_matrix,cbar=True,square=True,fmt=".1f",annot=True,annot_kws={'size':8},cmap="Blues")
#distribution plot
sns.distplot(df_new["price"],color="red")

#now defining the predictors and the target columns and doing the train_test split


predictors = df_new.loc[:, df_new.columns!="price"]
type(predictors)

target = df_new["price"]
type(target)
df_new.columns
# Train Test partition of the data and perfoming the adaboost regressor as it has given best result in automl by pycaret
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20,random_state=2)

from sklearn.ensemble import AdaBoostRegressor as AR
from sklearn import metrics
regressor=AR(base_estimator=None,learning_rate=1.0,loss="linear",n_estimators=100,random_state=2)

regressor.fit(x_train,y_train)
#predicting a new result
y_pred=regressor.predict(x_test)
## accuracy score
from sklearn import metrics
r_square=metrics.r2_score(y_test, y_pred)
print(r_square)
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
print(mean_squared_log_error)


#plotting the actual price and the predicted price
plt.plot(y_test,color="blue",label="actual_price")
plt.plot(y_pred,color="red",label="predicted_price")
plt.title("Actual_price vs Predicted_price")
plt.xlabel("values")
plt.ylabel("price")
plt.legend()
plt.show()
#save the model_ar to the disk
filename="model_ar.pkl"
pickle.dump(regressor,open(filename,"wb"))
model_ar=pickle.load(open("model_ar.pkl","rb"))

