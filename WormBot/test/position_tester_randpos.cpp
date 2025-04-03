// #include <Arduino.h>
#include <AccelStepper.h>
#include <SPI.h>
#include <SD.h>

File logFile;

#define BASE_SPEED 200
int SPEED = 200;

#define POS_0 -400
#define POS_1 0

int goal_pos_0 = POS_0;
int goal_pos_1 = POS_1;

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

  // Declare pins as Outputs
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(inputPin, INPUT);

  stepper.setCurrentPosition(0);
  stepper.setMaxSpeed(SPEED);   // the motor accelerates to this speed exactly without overshoot. Try other values.
  stepper.moveTo(POS_0); // CALL BEFORE SETTING SPEED!!11!1111!
  stepper.setSpeed(SPEED);

  Serial.begin(500000);
  while (!Serial);
  while (Serial.available() == 0);
  
  if (Serial.available() > 0) {
    String input = Serial.readString();
    int index = input.indexOf("cycles=");
    if (index != -1) {
      int receivedCycles = input.substring(index + 7).toInt();
      if (receivedCycles > 0) {
      CYCLES = receivedCycles;
      }
    }
  }
}


void loop()
{ 
  stepper.runSpeedToPosition();
  report();
  if (stepper.currentPosition() == goal_pos_0)
  {
    goal_pos_1 = POS_1 - random(0,200);
    stepper.moveTo(goal_pos_1);
    stepper.setSpeed(BASE_SPEED);
    uint64_t time=millis();
    while(millis()-time<1000){
      report();
    }
  }else if (stepper.currentPosition() == goal_pos_1)
  {
    goal_pos_0 = POS_0 + random(0,200);
    stepper.moveTo(goal_pos_0);
    stepper.setSpeed(BASE_SPEED);
    uint64_t time=millis();
    while(millis()-time<1000){
      report();
    }
    cycle++;
    if (cycle == CYCLES)
    {
      stepper.moveTo(0);
      stepper.setSpeed(SPEED);
      // stepper.setAcceleration(SPEED);
      // stepper.runToPosition();
      while (1)
      {
        stepper.runSpeedToPosition();
        report();
        if(stepper.currentPosition()==0){
          break;
        }
      }
      // reboot
      // logFile.close();
      SRC_GPR5 = 0x0BAD00F1;
      SCB_AIRCR = 0x05FA0004;
      while (1) ;
      
      }
    
  }
  report();

}


void report()
{
  sum += analogRead(inputPin);
  count++;
  if ( millis()-last_report>REPORT_INTERVAL)
  {
    last_report=millis();
    Serial.println(String(cycle)+":"+String(stepper.currentPosition())+":"+String(sum/count));
    // logFile.println(String(cycle)+":"+String(stepper.currentPosition())+":"+String(sum/count));
    sum = 0;
    count = 0;
  }
}