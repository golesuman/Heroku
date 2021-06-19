from logging import Manager
from operator import mod
from pandas.io.sql import pandasSQL_builder
import streamlit as st
import pandas as pd
st.write("""
# Simple Car Class Classifier App

## This App predicts the Car Class !
""")
st.sidebar.header('User input Paramers:')

def user_input_features():
    Buying=st.sidebar.slider("Buying",1.0,1.5,3.0)
    Maint=st.sidebar.slider("Maint",1.0,2.5,3.0)
    Doors=st.sidebar.slider("Doors",1.0,2.0,3.0)
    Persons=st.sidebar.slider("Persons",1.0,1.8,2.0)
    Lug_boot=st.sidebar.slider("Lug_boot",1.0,1.6,2.0)
    Safety=st.sidebar.slider("Safety",1.0,1.3,2.0)
    data={
        'Buying':Buying,
        'Maint':Maint,
        'Doors':Doors,
        'Persons':Persons,
        'Lug_boot':Lug_boot,
        'Safety':Safety
    }
    features=pd.DataFrame(data,index=[0])
    return features
df_features=user_input_features()
st.subheader("User Input Parameters")
st.write(df_features)

df_num=pd.read_csv("car.csv")
df=df_num.copy()
df.columns=['Buying','Maint','Doors','Persons','Lug_boot','Safety','class']
st.subheader('Class labels and their corresponding index number')
st.write(df['class'].unique())
lis_category=[feature for feature in df.columns if df[feature].dtypes=='O']
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
for feature in lis_category:
    df[feature]=lab.fit_transform(df[feature])
df_1=df.drop(['class'],axis=1)
x=df_1.loc[:1500]
df_y=df['class']
y=df_y.loc[:1500]
params={'C':[10,20,30,40],
    'kernel':['rbf','linear']
    }
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(gamma='auto'),params,cv=5,return_train_score=False)
model=SVC(C=40,kernel='rbf',gamma='auto')
model.fit(x,y)
import pickle 
pickle.dump(model,open('car_model.pkl','wb'))
prediction=model.predict(df_features)
# prediction_proba=model.predict_proba(df_features)
st.subheader('Prediction')
c=df['class']
st.write(prediction)