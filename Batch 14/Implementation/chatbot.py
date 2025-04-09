import streamlit as st
import requests
import speech_recognition as sr
import pyttsx3

# 🚨 Replace this with your actual Gemini API Key
API_KEY = "AIzaSyBOzA_Z2Dm9YzSKDmpCjKj72llZjZWalxU"

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Gemini API endpoint
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
headers = {"Content-Type": "application/json"}

# Text-to-Speech function
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech-to-Text function
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak now.")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ Could not understand your voice.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition service not available.")
    return None

# Ask Gemini API
def ask_gemini(prompt):
    full_prompt = f"""
You are a friendly, elderly-focused medical assistant. 
Give clear, simple advice for symptoms, first aid, or when to seek medical help.

User: {prompt}
"""
    data = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ]
    }
    response = requests.post(f"{GEMINI_URL}?key={API_KEY}", headers=headers, json=data)
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Sorry, I couldn't process that. Please try again."

# Streamlit UI
st.set_page_config(page_title="Voice Medical Assistant", page_icon="🩺")
st.title("🩺 Elderly Medical Chatbot (Voice Based)")
st.markdown("Speak your health concerns and get instant advice.")

if st.button("🎙️ Speak Now"):
    user_input = listen()
    if user_input:
        response = ask_gemini(user_input)
        st.markdown("### 🤖 Assistant says:")
        st.write(response)
        speak(response)
