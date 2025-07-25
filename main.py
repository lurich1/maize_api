import os
import base64
import requests
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o"  # Using OpenRouter's model notation

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_origins=["*"],
)

class ImageResponse(BaseModel):
    suggestions: str

class ChatbotResponse(BaseModel):
    response: str

class WeatherInsightsResponse(BaseModel):
    insights: str

def encode_image(image_file):
    return base64.b64encode(image_file).decode("utf-8")

def make_openrouter_request(messages, temperature=0.0):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Maize Disease Detection API",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature
    }
    
    response = requests.post(
        url=OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload)
    )
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

def generate_suggestions(image_base64):
    return make_openrouter_request([
        {
            "role": "system",
            "content": """You are a helpful maize doctor that provides suggestions based on images of maize plants.
            You should only detect and analyze maize plants and the following specific conditions:
            - Fall armyworm
            - Grasshopper damage
            - Healthy maize plants
            - Leaf beetle damage
            - Leaf blight
            - Leaf spot
            - Streak virus
            
            If the image is not of a maize plant, respond with: 'This appears to not be a maize plant. Please upload an image of a maize plant for analysis.'"""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this maize plant and identify any issues from the specified classes."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ])

@app.post("/analyze-maize-plant", response_model=ImageResponse)
async def analyze_maize_plant(file: UploadFile = File(...)):
    image_data = await file.read()
    image_base64 = encode_image(image_data)
    suggestions = generate_suggestions(image_base64)
    return JSONResponse(content={"suggestions": suggestions})

def generate_chatbot_response(user_input):
    return make_openrouter_request([
        {
            "role": "system",
            "content": "You are an agriculture expert chatbot that provides advice and information to maize farmers only. Don't respond to anything outside the context of maize farming."
        },
        {
            "role": "user",
            "content": user_input
        }
    ])

@app.post("/maize-chatbot", response_model=ChatbotResponse)
async def maize_chatbot(query: str = Form(...)):
    chatbot_response = generate_chatbot_response(query)
    return JSONResponse(content={"response": chatbot_response})

# Weather recommendation functions (all adapted for OpenRouter)
def generate_overall_recommendation(temperature, humidity, windspeed, pressure):
    return make_openrouter_request([
        {
            "role": "system",
            "content": "You are an agriculture expert that provides overall recommendations for maize farmers based on weather data."
        },
        {
            "role": "user",
            "content": f"The current weather conditions are: Temperature: {temperature}°C, Humidity: {humidity}%, Wind Speed: {windspeed} m/s, Pressure: {pressure} hPa."
        }
    ])

@app.post("/overall-recommendation", response_model=WeatherInsightsResponse)
async def overall_recommendation(
    temperature: float = Form(...),
    humidity: float = Form(...),
    windspeed: float = Form(...),
    pressure: float = Form(...),
):
    insights = generate_overall_recommendation(temperature, humidity, windspeed, pressure)
    return JSONResponse(content={"insights": insights})

# [Include all your other weather endpoints here with the same pattern]
# They all use make_openrouter_request() just like generate_overall_recommendation()

