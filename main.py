import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware  

model=joblib.load('Mental_Health_Model.pkl')


app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# a first pydentic Model to validate the input data
class StudentData(BaseModel):
    Age: int= Field(..., gt=0,le=100)
    Gender: str= Field(..., regex='^(Male|Female|Other)$')
    Country: str= Field(..., regex='^(India|USA|Canada|Australia|UK|Germany|Mexico|Turkey|France|Other)$')
    Academic_Level: Literal['High School', 'Undergraduate', 'Graduate']
    Most_Used_Platform: Literal['Instagram', 'Facebook', 'Twitter', 'Snapchat', 'TikTok', 'YouTube', 'Reddit', 'LINE', 'WhatsApp', 'WeChat', 'kakaoTalk', 'Vkontakte']
    Purpose_Of_Use: Literal['Entertainment', 'Education', 'Socializing', 'Work']
    Avg_Daily_Usage_Hours: float= Field(..., gt=0,le=24)
    Daily_Unlocks: int= Field(..., ge=0)
    Study_Hours: float= Field(..., ge=0,le=24)
    Physical_Activity_Hours: float= Field(..., ge=0,le=24)
    Sleep_Hours_Per_Night: float= Field(..., ge=0,le=24)
    Stress_Level: Literal['Low', 'Medium', 'High']
    Mental_Health_Score: float= Field(..., ge=0, le=10)
    
    
    
class predictionResponse(BaseModel):
    prediction_mental_health_score: float
    
     

@app.get('/')
def greet():
    return{"message": "Hello, World!"}
top_countries=['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France',]

@app.post ('/predict',response_model=predictionResponse)
def predict(data: StudentData):
    Country_Grouping = data.Country.apply(lambda x: x if x in top_countries else 'Other')
    
    input_row=pd.DataFrame([{
        'Age': data.Age,
        'Gender': data.Gender,
        'Country': data.Country,
        'Academic_Level': data.Academic_Level,
        'Most_Used_Platform': data.Most_Used_Platform,
        'Purpose_Of_Use': data.Purpose_Of_Use,
        'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks': data.Daily_Unlocks,
        'Study_Hours': data.Study_Hours,
        'Physical_Activity_Hours': data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,
        'Stress_Level': data.Stress_Level,
        'Mental_Health_Score': data.Mental_Health_Score,
        'Grouped_Country': Country_Grouping
    }])
    
    prediction = model.predict(input_row)[0]
    return predictionResponse(prediction_mental_health_score=round(float(prediction), 2))