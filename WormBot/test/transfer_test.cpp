// #include <Arduino.h>
#include <AccelStepper.h>
#include <SPI.h>
#include <SD.h>


struct{
    u_int32_t time1;
    u_int32_t time2;
    u_int32_t time3;
} times;

void setup()
{
  Serial.begin(500000);
}


void loop()
{ 
    while (!Serial.available());
    while (Serial.available())
    {
        Serial.read();
    }
    times.time1 = times.time2 = times.time3 = millis();

    // unsigned long currentMillis = millis();
    Serial.write((const uint8_t*)&times, sizeof(times));
    Serial.println();
    Serial.flush();
}