#include <AccelStepper.h>
 
// AccelStepper stepper; // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5
const int dirPin = 2;
const int stepPin = 3;


AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);
 
void setup()
{  
   stepper.setMaxSpeed(1000);
   stepper.move(1000);
   stepper.setSpeed(500);
   
}
 
void loop()
{  
   stepper.runSpeedToPosition();
}