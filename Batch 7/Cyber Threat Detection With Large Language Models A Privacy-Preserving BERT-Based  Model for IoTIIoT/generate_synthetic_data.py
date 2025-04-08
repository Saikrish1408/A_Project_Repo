import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random
import csv

# Set up the path for saving synthetic data
output_file_path = 'C:/Users/DELL/OneDrive/Desktop/project/synthetic_traffic_data.csv'

# Load pre-trained BERT model and tokenizer
model_path = "C:/Users/DELL/OneDrive/Desktop/project/bert_cyber.pth"  # Path to the pre-trained/fine-tuned BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Function to generate random network traffic data
def generate_synthetic_data():
    # Possible values for normal traffic
    protocols = ["TCP", "UDP", "ICMP"]
    states = ["ESTAB", "SYN_SENT", "CLOSED"]

    # Randomly choose protocol and state
    proto = random.choice(protocols)
    state = random.choice(states)

    # Normal traffic data generation (more common)
    if random.random() > 0.3:  # 70% for normal traffic
        sbytes = random.randint(100, 5000)
        dbytes = random.randint(100, 5000)
        spkts = random.randint(5, 50)
        dpkts = random.randint(5, 50)
        label = "Normal"
    # Attack traffic generation (more rare)
    else:  # 30% for attack traffic
        sbytes = random.randint(20000, 100000)
        dbytes = random.randint(50, 500)
        spkts = random.randint(1000, 5000)
        dpkts = random.randint(1, 20)
        label = "Attack"

    # Create an example input string (a form of traffic log)
    input_text = f"Protocol {proto} State {state} Bytes Sent {sbytes} Bytes Received {dbytes} Packets Sent {spkts} Packets Received {dpkts}"

    # Tokenize and make prediction
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        
    # Generate data entry (you could use the model's output here)
    data_entry = {
        "Protocol": proto,
        "State": state,
        "Bytes Sent": sbytes,
        "Bytes Received": dbytes,
        "Packets Sent": spkts,
        "Packets Received": dpkts,
        "Label": label,
        "Predicted Class": "Normal" if predicted_class == 0 else "Attack"
    }

    return data_entry

# Function to generate synthetic dataset and save to CSV
def generate_and_save_data(num_entries=100):
    # Open a file in write mode
    with open(output_file_path, mode='w', newline='') as file:
        fieldnames = ["Protocol", "State", "Bytes Sent", "Bytes Received", "Packets Sent", "Packets Received", "Label", "Predicted Class"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        # Generate synthetic data
        for _ in range(num_entries):
            data_entry = generate_synthetic_data()
            writer.writerow(data_entry)

        print(f"Generated {num_entries} synthetic data points and saved to {output_file_path}")

# Run the synthetic data generation and saving
generate_and_save_data(100)
