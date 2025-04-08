// Define the analog pin where the pulse sensor is connected
const int pulsePin = A0;  // A0 for pulse sensor

// Variables to store pulse sensor data
int pulseValue = 0;          // To store the pulse sensor value
int pulseRate = 0;           // To store the calculated pulse rate
int pulseThreshold = 550;    // Threshold value for detecting a heartbeat

// Variables for calculating pulse rate
unsigned long lastPulseTime = 0;
unsigned long pulseInterval = 0;
unsigned long currentMillis = 0;
unsigned long lastMillis = 0;

void setup() {
    Serial.begin(9600);
    pinMode(pulsePin, INPUT);  // Set the pulsePin as input
    Serial.println("Heartbeat Pulse Sensor Test");
}

void loop() {
    // Read the pulse sensor value
    pulseValue = analogRead(pulsePin);

    // Check if the pulse value is above the threshold (indicating a heartbeat)
    if (pulseValue > pulseThreshold) {
        currentMillis = millis();

        // Calculate the pulse rate if enough time has passed
        if (currentMillis - lastMillis > 500) { // Update pulse rate every 500ms
            pulseRate = 60000 / (currentMillis - lastMillis); // Calculate pulse rate (beats per minute)
            lastMillis = currentMillis;
        }

        // Output the pulse value and pulse rate to the Serial Monitor
        Serial.print("Pulse Value: ");
        Serial.print(pulseValue);
        Serial.print("\tPulse Rate: ");
        Serial.println(pulseRate);
    }

    // Optional: Add some delay to make the data more readable
    delay(100);
}
