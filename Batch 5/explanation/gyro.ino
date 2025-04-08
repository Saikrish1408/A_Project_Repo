#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
    Serial.begin(9600);
    Wire.begin();

    Serial.println("Initializing MPU6050...");
    mpu.initialize();

    if (mpu.testConnection()) {
        Serial.println("MPU6050 connected successfully!");
    } else {
        Serial.println("MPU6050 connection failed! Check wiring or I2C address.");
        while (1); // Halt execution
    }
}

void loop() {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    Serial.print("Accel X: "); Serial.print(ax);
    Serial.print(" | Y: "); Serial.print(ay);
    Serial.print(" | Z: "); Serial.println(az);

    Serial.print("Gyro X: "); Serial.print(gx);
    Serial.print(" | Y: "); Serial.print(gy);
    Serial.print(" | Z: "); Serial.println(gz);

    delay(500);
}
