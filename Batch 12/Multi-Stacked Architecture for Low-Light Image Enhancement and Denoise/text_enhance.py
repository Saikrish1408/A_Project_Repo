import google.generativeai as genai
import os

def text_corrector_llm(text):

    genai.configure(api_key="Your_API_Key")

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    queries = "Correct me the spelling :"+text
    print(queries)
    response = model.generate_content(queries)
    print(response.text)
    return response.text

def text_enhancer(text):

    genai.configure(api_key="Your_API_Key")

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    queries = "Enhane the sentance and give me one sentence as a response :"+text
    print(queries)
    response = model.generate_content(queries)
    print(response.text)
    return response.text

# text_enhancer("the cde is nor written propely")