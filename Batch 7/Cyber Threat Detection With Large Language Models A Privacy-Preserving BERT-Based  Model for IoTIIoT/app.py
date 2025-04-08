import streamlit
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#
# Load the model and map it to CPU
#model = torch.load("bert_cyber.pth", map_location=torch.device('cpu'))
model = torch.load("bert_cyber.pth", map_location=torch.device('cpu'), weights_only=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Move model to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define function to preprocess input data
def preprocess_input(proto, state, sbytes, dbytes, spkts, dpkts):
    text = (
        f"Network traffic observed with protocol '{proto}' and state '{state}'. "
        f"The source sent {sbytes} bytes and {spkts} packets, "
        f"while the destination received {dbytes} bytes and {dpkts} packets."
    )
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    return inputs

# Function to make predictions
def predict(proto, state, sbytes, dbytes, spkts, dpkts):
    inputs = preprocess_input(proto, state, sbytes, dbytes, spkts, dpkts)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    class_labels = {0: "Normal", 1: "Attack"}  # Update based on your labels
    return class_labels[predicted_class]

# Streamlit App UI
streamlit.title("🔍 Cyber Threat Detection with BERT")
streamlit.write("Enter network traffic details to predict whether it's normal or an attack.")

# User Inputs
proto = streamlit.selectbox("Protocol", ["tcp", "udp", "icmp"])
state = streamlit.selectbox("State", ["ESTAB", "SYN_SENT", "CLOSED"])
sbytes = streamlit.number_input("Bytes Sent", min_value=0, step=1)
dbytes = streamlit.number_input("Bytes Received", min_value=0, step=1)
spkts = streamlit.number_input("Packets Sent", min_value=0, step=1)
dpkts = streamlit.number_input("Packets Received", min_value=0, step=1)

# Prediction Button
if streamlit.button("Predict"):
    result = predict(proto, state, sbytes, dbytes, spkts, dpkts)
    streamlit.success(f"Prediction: **{result}**")

