#include <AccelStepper.h>
#include "packets.h"

#define BASE_SPEED 600

// Define pin connections & motor's steps per revolution
const int stepA = 2;
const int dirA = 3;
const int stepB = 4;
const int dirB = 5;
const int stepC = 6;
const int dirC = 7;
// const int stepD = 8;
// const int dirD = 9;

// AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);
AccelStepper stepperA(AccelStepper::DRIVER, stepA, dirA);
AccelStepper stepperB(AccelStepper::DRIVER, stepB, dirB);
AccelStepper stepperC(AccelStepper::DRIVER, stepC, dirC);
// AccelStepper stepperD(AccelStepper::DRIVER, stepD, dirD);

void setup()
{
  // Declare pins as Outputs
  pinMode(stepA, OUTPUT);
  pinMode(dirA, OUTPUT);
  pinMode(stepB, OUTPUT);
  pinMode(dirB, OUTPUT);
  pinMode(stepC, OUTPUT);
  pinMode(dirC, OUTPUT);
  // pinMode(stepD, OUTPUT);
  // pinMode(dirD, OUTPUT);

  stepperA.setMaxSpeed(BASE_SPEED);
  stepperB.setMaxSpeed(BASE_SPEED);
  stepperC.setMaxSpeed(BASE_SPEED);
  // stepperD.setMaxSpeed(BASE_SPEED);

  // stepperA.moveTo(1000);
  // stepperB.moveTo(1000);
  // stepperC.moveTo(1000);
  // stepperD.moveTo(1000);

  stepperA.setSpeed(BASE_SPEED);
  stepperB.setSpeed(BASE_SPEED);
  stepperC.setSpeed(BASE_SPEED);
  // stepperD.setSpeed(BASE_SPEED);

  Serial.begin(500000);
  while (!Serial);
}


void loop()
{ 
  if (Serial.available() >= sizeof(motordata))
  {
    motordata data;
    Serial.readBytes((char*)&data, sizeof(data));
    if(data.header[0] != 'S' || data.footer[0] != 'E')
    {
      while (Serial.available() > 0) {
        Serial.read();
      }
      return;
    }
    stepperA.moveTo(data.motorA);
    stepperB.moveTo(data.motorB);
    stepperC.moveTo(data.motorC);
    // stepperD.moveTo(data.motorD);
    stepperA.setSpeed(BASE_SPEED);
    stepperB.setSpeed(BASE_SPEED);
    stepperC.setSpeed(BASE_SPEED);
    // stepperD.setSpeed(BASE_SPEED);
  }
  stepperA.runSpeedToPosition();
  stepperB.runSpeedToPosition();
  stepperC.runSpeedToPosition();
  // stepperD.runSpeedToPosition();
}
