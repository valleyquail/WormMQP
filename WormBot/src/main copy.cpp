// #include <Arduino.h>
#include <AccelStepper.h>

// Define pin connections & motor's steps per revolution
const int dirPin = 2;
const int stepPin = 3;

AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

// const int stepsPerRevolution = 200;
// const int microDelay = 50;

void setup()
{
  // Declare pins as Outputs
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  stepper.setSpeed(100);
  stepper.setMaxSpeed(200.0);   // the motor accelerates to this speed exactly without overshoot. Try other values.
  stepper.setAcceleration(50.0); 
  // stepper.setSpeed(100);
  stepper.moveTo(10000); 
}

void loop()
{
  // stepper.runSpeed();
  stepper.run();
}