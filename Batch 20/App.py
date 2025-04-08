from flask import Flask, render_template, request, jsonify
import voice

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('doctor_login.html')

@app.route('/patient_login')
def patient_login():
    return render_template('patient_login.html')

@app.route('/patient')
def patient():
    return render_template('index.html')

@app.route('/redirect')
def redirect():
    return render_template('redirect.html')

@app.route('/doctor')
def doctor_dashboard():
    return render_template('doctor.html')

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json  # Get the JSON data from the request
    english_text = data.get('english_text')
    language = data.get('language')
    result = voice.english_to_tamil_voice(english_text, language) 
    return jsonify({'message': result}) 

if __name__ == '__main__':
    app.run(debug=True)