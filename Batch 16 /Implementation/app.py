import sqlite3
from flask import *

from flask_cors import CORS
import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication
app.secret_key = "your_secret_key_here"

DATABASE = "hospital.db"
UPLOAD_FOLDER = "static/check"  # Folder for saving uploaded images
MODEL_PATH = "adam.keras"

def handle_image_upload(uploaded_file):
    base_dir = r'C:\Users\SASIDHARAN.N\OneDrive\Desktop\final frontend\check'

    # Remove old folder if it exists
    old_folder = os.path.join(base_dir, 'one')
    if os.path.exists(old_folder):
        shutil.rmtree(old_folder)

    # Create new folder
    new_folder = os.path.join(base_dir, 'one')
    os.makedirs(new_folder, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(new_folder, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Prepare ImageDataGenerator
    testGenerator = ImageDataGenerator().flow_from_directory(
        base_dir, target_size=(320, 320), batch_size=1, shuffle=False
    )

    # Get predictions
    predictions = model.predict(testGenerator)

    # Get predicted class label
    predicted_classes = np.argmax(predictions, axis=1)

    return int(predicted_classes[0])
# Function to create tables
def create_tables():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS doctors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        specialization TEXT NOT NULL,
                        license_number TEXT UNIQUE NOT NULL,
                        contact_number TEXT NOT NULL,
                        office_address TEXT NOT NULL,
                        years_experience INTEGER NOT NULL,
                        consultation_fee REAL NOT NULL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        first_name TEXT NOT NULL,
                        last_name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        date_of_birth TEXT NOT NULL,
                        blood_group TEXT NOT NULL,
                        contact_number TEXT NOT NULL,
                        address TEXT NOT NULL,
                        emergency_contact_name TEXT NOT NULL,
                        emergency_contact_number TEXT NOT NULL)''')
    
    conn.commit()
    conn.close()

create_tables()  # Ensure tables exist when the app starts



@app.route("/signin", methods=["post"])
def signin():
    mail1 = request.form['email']
    password1 = request.form['password']
    role=request.form['role']
    con = sqlite3.connect('hospital.db')
    if role=="Doctor":
        data = con.execute(
        "select id,first_name from doctors where `email` = ? AND `password` = ?", (mail1, password1,)).fetchone()
        print(data)
        if data!=None:
            session['username'] = data[0]
            session["role"]="doctor"
            return redirect("user")
        else:
            return render_template("index.html")
    else:
        data = con.execute(
            "select id,first_name from patients where `email` = ? AND `password` = ?", (mail1, password1,)).fetchone()
        print(data)
        if data!=None:
            session['username'] = data[0]
            session["role"]="patients"
            return redirect("user")
        else:
            return render_template("index.html")


@app.route('/bookappointment')
def bookappointment():
    con = sqlite3.connect('hospital.db')
    data = con.execute("select * from doctors where id='%s'"%(session["username"])).fetchall()
    
    return render_template("app.html",data=data)

@app.route('/viewappointment')
def viewappointmentdr():
    if session["role"]=="doctor":
        con = sqlite3.connect('hospital.db')
        data = con.execute("select * from appointment where did='%s'"%(session["username"])).fetchall()
        return render_template("viewappoint.html",data=data)
    elif session["role"]=="patients":
        con = sqlite3.connect('hospital.db')
        data = con.execute("select * from appointment where pid='%s'"%(session["username"])).fetchall()
        return render_template("viewappoint.html",data=data)


@app.route('/admin')
def admin():
    return render_template("admin.html")

@app.route('/user')
def user():
    print(session["role"]=="patients")
    return render_template("user.html")

@app.route('/oldreports')
def oldreports():
    con = sqlite3.connect('hospital.db')
    data = con.execute("select * from patientdetails where pid='%s'"%(session["username"])).fetchall()
    return render_template("oldreports.html",data=data)


@app.route('/diagnose', methods=['POST'])
def diagnose():
    import os
    image = request.files.get('image')
    image1 = request.files.get('image1')
    i=request.form["id"]
    

    classname = [
          "Calculus",
          "Caries",
          "Gingivitis",
          "Hypodontia",
          "Tooth Discoloration",
          "Ulcers",
        ];
    predict=handle_image_upload(image)
    
    DISEASE_INFO = {
    "Calculus": {
        "symptoms": ["Hard deposits on teeth", "Bad breath", "Gum inflammation"],
        "suggestion": "Visit a dentist for scaling and proper oral hygiene."
    },
    "Caries": {
        "symptoms": ["Tooth decay", "Sensitivity to hot/cold", "Visible holes in teeth"],
        "suggestion": "Maintain good oral hygiene and consult a dentist for fillings or fluoride treatment."
    },
    "Gingivitis": {
        "symptoms": ["Swollen, bleeding gums", "Bad breath", "Tender gums"],
        "suggestion": "Use antiseptic mouthwash, brush and floss daily, and visit a dentist if symptoms persist."
    },
    "Hypodontia": {
        "symptoms": ["Missing one or more teeth", "Spacing issues in teeth"],
        "suggestion": "Consult a dentist for dental implants or orthodontic treatment if necessary."
    },
    "Tooth Discoloration": {
        "symptoms": ["Yellow, brown, or gray stains on teeth", "Surface roughness"],
        "suggestion": "Avoid staining foods, use whitening toothpaste, and consider professional cleaning or bleaching."
    },
    "Ulcers": {
        "symptoms": ["Painful sores in the mouth", "Red or white patches", "Difficulty eating"],
        "suggestion": "Use medicated mouthwash, avoid spicy foods, and consult a doctor if persistent."
    }
}
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Save image to static folder
    image_path1 = os.path.join("static/check", image1.filename)
    image1.save(image_path1)
    print(id,image.filename,classname[predict])
    try:
        cursor.execute('''INSERT INTO patientdetails (pid ,imagedetails ,diagonis)
                          VALUES (?, ?, ?)''',
                       (i,image.filename,classname[predict]))
        conn.commit()
        return jsonify({"image":classname[predict],"DISEASE_INFO":DISEASE_INFO[classname[predict]]})
        
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email or License Number already exists'}), 400
    finally:
        conn.close()



  

    
# Register a Doctor
@app.route('/register/doctor', methods=['POST'])
def register_doctor():
    data = dict(request.form)
    

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute('''INSERT INTO doctors (first_name, last_name, email, password, specialization, license_number, contact_number, office_address, years_experience, consultation_fee)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (data['first_name'], data['last_name'], data['email'], data["password"], data['specialization'], data['license_number'], data['contact_number'], data['office_address'], data['years_experience'], data['consultation_fee']))
        conn.commit()
        return render_template("index.html")
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email or License Number already exists'}), 400
    finally:
        conn.close()

# Register a Patient
@app.route('/register/patient', methods=['POST'])
def register_patient():
    data = dict(request.form)
    

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute('''INSERT INTO patients (first_name, last_name, email, password, date_of_birth, blood_group, contact_number, address, emergency_contact_name, emergency_contact_number)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (data['first_name'], data['last_name'], data['email'], data["password"], data['dob'], data['blood_group'], data['contact_number'], data['address'], data['emergency_name'], data['emergency_contact']))
        conn.commit()
        return render_template("index.html")
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 400
    finally:
        conn.close()

@app.route("/")
def home():
    """Render the index page."""
    return render_template("index.html")


@app.route("/login")
def reg():
    """Render the index page."""
    return render_template("register.html")


# Add an Appointment
@app.route("/add_appointment", methods=["POST"])
def add_appointment():
    data = request.json
    did = data.get("did")
    pid = data.get("pid")
    appointment_date = data.get("appointmentdate")
    status = data.get("status", "Scheduled")  # Default status

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO appointment (did, pid, appointmentdate, status) VALUES (?, ?, ?, ?)",
        (did, pid, appointment_date, status),
    )
    conn.commit()
    conn.close()

    return jsonify({"message": "Appointment added successfully"}), 201


# View Appointments by Patient ID
@app.route("/appointments/patient/<int:pid>", methods=["GET"])
def view_appointments_by_patient(pid):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointment WHERE pid = ?", (pid,))
    columns = [col[0] for col in cursor.description]  # Get column names
    appointments = [dict(zip(columns, row)) for row in cursor.fetchall()]  # Convert to dict
    conn.close()
    return jsonify(appointments)

# View Appointments by Doctor ID
@app.route("/appointments/doctor/<int:did>", methods=["GET"])
def view_appointments_by_doctor(did):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointment WHERE did = ?", (did,))
    columns = [col[0] for col in cursor.description]  # Get column names
    appointments = [dict(zip(columns, row)) for row in cursor.fetchall()]  # Convert to dict
    conn.close()
    return jsonify(appointments)
if __name__ == '__main__':
    app.run()
