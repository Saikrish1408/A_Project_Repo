import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import io

# AES encryption/decryption functions using PyCryptodome
def encrypt_data(data, key):
    # Generate a random IV for each encryption (16 bytes for AES)
    iv = get_random_bytes(16)

    # Initialize AES cipher in CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Pad the data to ensure it is a multiple of 16 bytes
    padded_data = pad(data.encode(), AES.block_size)

    # Encrypt the data
    encrypted_data = cipher.encrypt(padded_data)

    # Return the IV + encrypted data, base64 encoded
    return base64.b64encode(iv + encrypted_data).decode()

def decrypt_data(encrypted_data, key):
    # Decode the base64 encrypted data
    encrypted_data = base64.b64decode(encrypted_data.encode())

    # Extract the IV and the encrypted data
    iv = encrypted_data[:16]  # First 16 bytes are the IV
    encrypted_data = encrypted_data[16:]  # The rest is the actual encrypted data

    # Initialize AES cipher with the IV
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt and unpad the data
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    # Return the decrypted text
    return decrypted_data.decode()

# Load your trained model (.h5)
def load_trained_model():
    model_path = 'model.h5'  # Replace with your actual model file path
    return load_model(model_path)

# Streamlit UI components
st.title('CSV File Prediction App with AES Encryption (PyCryptodome)')

# AES Key (must be 16, 24, or 32 bytes long for AES)
aes_key = b'1234567890123456'  # For demonstration, use a simple 16-byte key (use a stronger key in production)

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Encrypt the CSV file content
    encrypted_csv = encrypt_data(df.to_csv(index=False), aes_key)

    # Save the encrypted data to a local file
    encrypted_filename = 'encrypted_file.enc'
    with open(encrypted_filename, 'w') as f:
        f.write(encrypted_csv)
    
    st.write("Encrypted CSV has been saved locally.")
    # st.write("Encrypted CSV (Base64):", encrypted_csv) 

    # Show the DataFrame to the user (optional, plain text or decrypted)
    st.subheader('Uploaded DataFrame')
    st.write(df)

    # Load the trained model
    model = load_trained_model()

    # Make predictions when the button is pressed
    if st.button('Make Predictions'):
        # Make predictions using the loaded model
        predictions = model.predict(df) 
        predictions = predictions.round().astype(int)
        # Show predictions
        st.subheader('Predictions')
        st.text("Attack Detected" if predictions[0]==1 else "No Attack Detected")

        # Encrypt predictions before displaying (optional, depends on your use case)
        encrypted_predictions = encrypt_data(str(predictions), aes_key)
        st.write("Encrypted Predictions (Base64):", encrypted_predictions)

    # Provide a button to download the encrypted file
    st.subheader("Download Encrypted File")

    st.download_button(
                label="Download Encrypted CSV",
                data=open(encrypted_filename, 'rb'),
                file_name=encrypted_filename,
                mime="application/octet-stream"
            )

    # Provide a button to decrypt the encrypted file and show/download
    st.subheader("Decrypt and Download Decrypted File")

    if st.button('Decrypt and Show CSV'):
        # Decrypt the encrypted file content
        with open(encrypted_filename, 'r') as f:
            encrypted_data = f.read()
        
        decrypted_csv = decrypt_data(encrypted_data, aes_key)
        decrypted_df = pd.read_csv(io.StringIO(decrypted_csv))
        
        # Display the decrypted DataFrame
        st.subheader("Decrypted DataFrame")
        st.write(decrypted_df)

        # Provide an option to download the decrypted CSV file
        decrypted_filename = 'decrypted_file.csv'
        decrypted_df.to_csv(decrypted_filename, index=False)

        with open(decrypted_filename, 'rb') as f:
            st.download_button(
                label="Download Decrypted CSV",
                data=f,
                file_name=decrypted_filename,
                mime="text/csv"
            )