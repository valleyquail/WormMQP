// #include <Arduino.h>
#include <AccelStepper.h>
#include <SPI.h>
#include <SD.h>

File logFile;

#define SPEED 200

#define POS_0 -400
#define POS_1 0

// #define CYCLES 5
int CYCLES = 100;

// Define pin connections & motor's steps per revolution
const int dirPin = 2;
const int stepPin = 3;

//analog input pin
const int inputPin = A0;

AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

u_int16_t cycle = 0;
long last = 0;
long sum = 0;
double count = 0;

uint64_t last_report = 0;
#define REPORT_INTERVAL 25

void report();

void setup()
{
  Serial.begin(115200);

  pinMode(inputPin, INPUT);


}


void loop()
{ 
  report();
}


void report()
{
  sum += analogRead(inputPin);
  count++;
  if ( millis()-last_report>REPORT_INTERVAL)
  {
    last_report=millis();
    // Serial.print("C: ");
    // Serial.print(cycle);
    // Serial.print(" T: ");
    // Serial.print(millis());
    // Serial.print(" P: ");
    // Serial.print(stepper.currentPosition());
    // Serial.print(" V: ");
    // Serial.println(sum/count);
    // Serial.println("C: "+String(cycle)+" P: "+String(stepper.currentPosition())+" V: "+String(sum/count));
    // Serial.println(String(cycle)+":"+String(stepper.currentPosition())+":"+String(sum/count));
    Serial.println(sum/count);
    // logFile.println(String(cycle)+":"+String(stepper.currentPosition())+":"+String(sum/count));
    sum = 0;
    count = 0;
  }
}