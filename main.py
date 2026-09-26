import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("Mental_Health_Model.pkl")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TOP_COUNTRIES = ["Other", "India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "Turkey", "France"]


class StudentData(BaseModel):
    Age: int = Field(..., gt=0, le=100)
    Gender: Literal["Male", "Female", "Other"]
    Country: str  # any raw country string is accepted; grouped below
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Literal[
        "Instagram", "TikTok", "Facebook", "LinkedIn", "YouTube",
        "Twitter", "Snapchat", "WhatsApp", "LINE", "VKontakte",
        "KakaoTalk", "WeChat",
    ]
    Purpose_Of_Use: Literal["Entertainment", "Education", "Socializing", "Work", "Networking"]
    Avg_Daily_Usage_Hours: float = Field(..., gt=0, le=24)
    Daily_Unlocks: int = Field(..., ge=0)
    Study_Hours: float = Field(..., ge=0, le=24)
    Physical_Activity_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]
    # NOTE: Mental_Health_Score deliberately removed — it's the target
    # we're predicting, not an input the user should ever provide.


class PredictionResponse(BaseModel):
    prediction_mental_health_score: float


@app.get("/")
def greet():
    return {"message": "Hello, World!"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    # `data.Country` is a single string, not a Series -> no .apply() here.
    grouped_country = data.Country if data.Country in TOP_COUNTRIES else "Other"

    input_row = pd.DataFrame(
        [
            {
                "Age": data.Age,
                "Gender": data.Gender,
                "Academic_Level": data.Academic_Level,
                "Most_Used_Platform": data.Most_Used_Platform,
                "Purpose_Of_Use": data.Purpose_Of_Use,
                "Avg_Daily_Usage_Hours": data.Avg_Daily_Usage_Hours,
                "Daily_Unlocks": data.Daily_Unlocks,
                "Study_Hours": data.Study_Hours,
                "Physical_Activity_Hours": data.Physical_Activity_Hours,
                "Sleep_Hours_Per_Night": data.Sleep_Hours_Per_Night,
                "Stress_Level": data.Stress_Level,
                # Must match the exact column name the pipeline was
                # trained on: "Grouped_country" (lowercase c).
                "Grouped_country": grouped_country,
            }
        ]
    )

    try:
        prediction = model.predict(input_row)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(prediction_mental_health_score=round(float(prediction), 2))
