// #include <Arduino.h>
#include <AccelStepper.h>

// Define pin connections & motor's steps per revolution
const int dirPin = 2;
const int stepPin = 3;

//analog input pin
const int inputPin = A0;

AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

// const int stepsPerRevolution = 200;
// const int microDelay = 50;

void setup()
{
  // Declare pins as Outputs
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(inputPin, INPUT);

  
  stepper.setMaxSpeed(600.0);   // the motor accelerates to this speed exactly without overshoot. Try other values.
  stepper.setAcceleration(600.0); 
  stepper.setSpeed(600);
  stepper.moveTo(3000); 

  Serial.begin(115200);
}


long last = 0;
long sum = 0;
double count = 0;
void loop()
{ 
  while(stepper.run()){
  // stepper.runSpeed();
  sum += analogRead(inputPin);
  count++;

  if (millis() - last > 100)
  {
    Serial.println(sum/count);
    sum = 0;
    count = 0;
    last = millis();
  }
}
  stepper.moveTo(0);
  while(stepper.run()){
  // stepper.runSpeed();
  sum += analogRead(inputPin);
  count++;

  if (millis() - last > 100)
  {
    Serial.println(sum/count);
    sum = 0;
    count = 0;
    last = millis();
  }
}
stepper.moveTo(3000);
}