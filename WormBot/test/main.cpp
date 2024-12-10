#include <Arduino.h>

// // put function declarations here:
// int myFunction(int, int);

// void setup() {
//   // put your setup code here, to run once:
//   int result = myFunction(2, 3);
// }

// void loop() {
//   // // put your main code here, to run repeatedly:
//   // digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
//   // delay(1000);                       // wait for a second
//   // digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//   // delay(1000);                       // wait for a second


// }

// // put function definitions here:
// int myFunction(int x, int y) {
//   return x + y;
// }
// Define pin connections & motor's steps per revolution
const int dirPin = 2;
const int stepPin = 3;
// const int leftPin = A0;
// const int rightPin = A1;
const int stepsPerRevolution = 200;

const int microDelay = 50;

void setup()
{
  // Declare pins as Outputs
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);

  // pinMode(leftPin, INPUT_PULLUP);
  // pinMode(rightPin, INPUT_PULLUP);
}
void loop()
{
  // if(!digitalRead(leftPin)){
    digitalWrite(dirPin, LOW);
    
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(microDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(microDelay);
  // }else if(!digitalRead(rightPin)){
  //   digitalWrite(dirPin, HIGH);
     
  //   digitalWrite(stepPin, HIGH);
  //   delayMicroseconds(microDelay);
  //   digitalWrite(stepPin, LOW);
  //   delayMicroseconds(microDelay);
  // }
}