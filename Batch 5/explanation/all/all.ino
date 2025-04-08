 #include <Wire.h>
#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

// TFT Display
MCUFRIEND_kbv tft;

// Color definitions
#define BLACK   0x0000
#define BLUE    0x001F
#define RED     0xF800
#define GREEN   0x07E0
#define CYAN    0x07FF
#define MAGENTA 0xF81F
#define YELLOW  0xFFE0
#define WHITE   0xFFFF

// Pulse rate parameters
int pulseRate = 50;  // Default pulse rate (between 40-65)
int pulseY = 80;     // Initial Y position for the heartbeat line
int pulseDirection = 1; // Pulse direction (1 for moving down, -1 for reset)

// Graph boundaries
int graphX = 20, graphY = 80, graphWidth = 100, graphHeight = 120;

void setup() {
    // Initialize TFT
    uint16_t ID = tft.readID();
    if (ID == 0xD3D3) ID = 0x9486; // Default ID for MCUFRIEND
    tft.begin(ID);
    tft.setRotation(1); // Landscape orientation
    tft.fillScreen(BLACK);

    // Draw the static layout
    drawDummyLayout();
}

void loop() {
    // Simulate heartbeat by updating the graph and pulse rate display
    simulateHeartbeat();
    displayPulseRate();
    delay(100); // Update the heartbeat every 100 ms (adjust as needed)
}

void drawDummyLayout() {
    // Title: "GuardianEye"
    tft.setTextColor(YELLOW, BLACK);
    tft.setTextSize(2); // Adjusted text size for better fit
    int headingWidth = 10 * 13;  // Using font width of 6 pixels for size 2 text
    tft.setCursor((240 - headingWidth) / 2, 10); // Centered heading on 240x320 screen
    tft.print("GuardianEye");

    // Column 1 - Heartbeat and Graph (Including Heart Logo)
    tft.setTextSize(2);
    tft.setTextColor(WHITE, BLACK);
    tft.setCursor(20, 50);
    tft.print("Heartbeat");
    tft.drawRect(graphX, graphY, graphWidth, graphHeight, WHITE); // Adjusted graph size for better fit

    // Draw heart logo inside the graph
    drawHeartInsideRect(graphX + 25, graphY + 25);  // Positioning the heart within the graph box

    // Column 2 - Temperature, Humidity, Shake, and Fall
    tft.setTextSize(2);
    tft.setTextColor(WHITE, BLACK);
    tft.setCursor(140, 50);
    tft.print("Temp: 25.0 C"); // Dummy temperature
    tft.setCursor(140, 80);
    tft.print("Hum: 60.0 %");  // Dummy humidity

    // Shake and Fall Detection
    tft.setTextSize(2); // Smaller text size for buttons
    tft.setCursor(130, 150);
    tft.setTextColor(RED, BLACK);
    tft.print("Shake?: No");

    tft.setCursor(130, 170);
    tft.print("Fall?: No");
}

void simulateHeartbeat() {
    // Redraw the background of the graph to prevent overlap with the heartbeat line
    tft.fillRect(graphX + 1, graphY + 1, graphWidth - 2, graphHeight - 2, BLACK);

    // Redraw the heart inside the graph (to ensure it persists)
    drawHeartInsideRect(graphX + 25, graphY + 25);

    // Draw the heartbeat line inside the graph
    tft.drawLine(graphX + 1, pulseY, graphX + graphWidth - 1, pulseY, GREEN); // Draw heartbeat line (vertical line)

    // Update the Y position of the heartbeat line
    pulseY += pulseDirection;

    // When the line reaches the bottom, reset and start from the top
    if (pulseY >= graphY + graphHeight - 1) {
        pulseDirection = -1; // Start moving up
    }
    if (pulseY <= graphY + 1) {
        pulseDirection = 1; // Start moving down again
    }
}

void displayPulseRate() {
    // Display the pulse rate below the heartbeat graph
    tft.setTextSize(1); // Smaller text size for pulse rate display
    tft.setTextColor(WHITE, BLACK);
    tft.setCursor(graphX, graphY + graphHeight + 5); // Position below the heartbeat graph
    tft.print("Pulse Rate: ");
    
    // Display the current pulse rate (random between 40 and 65)
    pulseRate = random(40, 66); // Generate a random pulse rate between 40 and 65
    tft.setTextColor(RED, BLACK); // Red color for pulse rate
    tft.print(pulseRate);
}

void drawHeartInsideRect(int x, int y) {
    // Heart shape with bigger circles and intersecting
    int heartWidth = 60;  // Further increase heart width
    int heartHeight = 60; // Further increase heart height
    
    // Left and right top curves of the heart (using circles)
    tft.fillCircle(x + 15, y + 17, 15, RED);   // Left curve, larger radius
    tft.fillCircle(x + 35, y + 17, 15, RED);   // Right curve, larger radius

    // Bottom part of the heart (using a triangle)
    tft.fillTriangle(x + 5, y + 30, x + 45, y + 30, x + 25, y + 50, RED); // Bottom part of the heart
}



