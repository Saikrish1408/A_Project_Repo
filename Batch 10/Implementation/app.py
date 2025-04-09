from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os

app = Flask(__name__)

# 1. Securely get API key from environment variable
api_key = "AIzaSyBEWbGKJ5tOFjTdfbZ8wRMz1G5DyLV5U9E"

# 2. Configure genai with the retrieved key
genai.configure(api_key=api_key)

# 3. Set generation config (consistent with your other code)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# 4. Initialize the model with the config
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",  # Or "gemini-pro" if needed
    generation_config=generation_config,
)

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/chat_page')
def chat_page():
    return render_template('index.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # 5. Use start_chat and send_message for chat history (if needed)
    chat_session = model.start_chat(history=[])  # Initialize chat history
    response = chat_session.send_message("give me the response as a plain text in a precise manner don't make any text styling and maintian the response in a medical context if the questin is like refer or give somre doctor or specialist near me then suggest only the doctor names their contact their address,  reviews no any other data is needed give me only that if doctor details is asked so the user query is: " + user_message) # Send message to current chat session

    return jsonify({'bot_message': response.text})

if __name__ == '__main__':
    app.run(debug=True)