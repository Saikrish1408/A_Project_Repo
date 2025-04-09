from flask import *
import sqlite3

import os



# Function to handle file upload, process and delete old folder
def handle_image_upload(uploaded_file):
    import shutil
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    base_dir = r'C:\Users\Home\Downloads\caries_classification\check'

    # Remove the existing folder if it exists
    old_folder = os.path.join(base_dir, 'one')
    if os.path.exists(old_folder):
        shutil.rmtree(old_folder)
    
    # Create a new folder to store the uploaded file
    new_folder = os.path.join(base_dir, 'one')
    os.makedirs(new_folder)

    # Save the uploaded file in the new folder
    uploaded_file.save(os.path.join(new_folder, uploaded_file.filename))

    # Load the trained model
    model = tf.keras.models.load_model('adam.keras')

    # Initialize ImageDataGenerator
    testGenerator = ImageDataGenerator().flow_from_directory(
       base_dir, target_size=(320, 320), batch_size=32, shuffle=False
    )

    # Get predictions
    predictions = model.predict(testGenerator)

    # Get predicted class labels
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes


# Example of usage:

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Ensure the file is valid
    filename = secure_filename(file.filename)
    
    # Process the uploaded image and get predictions
    predicted_classes = handle_image_upload(file)
    
    return jsonify({"predicted_classes": predicted_classes.tolist()})




@app.route("/login")
def login():
    return render_template("login.html")


@app.route('/logon')
def logon():
    return render_template('signup.html')


@app.route("/signup", methods=["post", "get"])
def signup():
    username = request.form['user']
    name = request.form['name']
    email = request.form['email']
    number = request.form["mobile"]
    password = request.form['password']
    role = request.form['role']
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`,'role') VALUES (?, ?, ?, ?, ?,?)",
                (username, email, password, number, name, role))
    con.commit()
    con.close()
    return render_template("index.html")


@app.route("/signin", methods=["post"])
def signin():
    mail1 = request.form['user']
    password1 = request.form['password']
    con = sqlite3.connect('signup.db')
    data = 0
    data = con.execute(
        "select `user`, `mobile`,role from info where `user` = ? AND `password` = ?", (mail1, password1,)).fetchone()
    print(data)
    try:
        if data[2]=="Doctor":
                session['username'] = data[0]
                session['mobile'] = data[1]
                return redirect("admin")
        elif data[2]=='User':
            session['username'] = data[0]
            session['mobile'] = data[1]
            return redirect("check")
    except Exception as e:
         print(e)
         return render_template("signup.html")

@app.route('/admin')
def admin():
    return render_template("admin.html")
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/check')
def index():
    return render_template("check.html")



if __name__ == "__main__":
    app.run(debug=True)
