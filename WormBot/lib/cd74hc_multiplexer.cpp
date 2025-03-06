//
// Created by nikesh on 2/17/25.
//

#include "cd74hc_multiplexer.h"
#include <Arduino.h>
#include <../include/ADC/ADC.h>


CD74HC_Multiplexer::CD74HC_Multiplexer(uint8_t mux0, uint8_t mux1, uint8_t mux2,uint8_t mux3,
                                       uint8_t analog_pin, uint8_t num_sensors,
                                       ADC *adc) {
    this->__mux0 = mux0;
    this->__mux1 = mux1;
    this->__mux2 = mux2;
    this->__mux3 = mux3;
    this->__analog_pin = analog_pin;
    this->__num_sensors = num_sensors;
    this->__adc = adc;
    pinMode(this->__mux0, OUTPUT);
    pinMode(this->__mux1, OUTPUT);
    pinMode(this->__mux2, OUTPUT);
    pinMode(this->__mux3, OUTPUT);
    pinMode(this->__analog_pin, INPUT);
    this->__data_buff = new uint16_t[num_sensors];
}

void CD74HC_Multiplexer::selectChannel(uint8_t channel) {
    digitalWrite(__mux0, bitRead(channel, 0));
    digitalWrite(__mux1, bitRead(channel, 1));
    digitalWrite(__mux2, bitRead(channel, 2));
    digitalWrite(__mux3, bitRead(channel, 3));
}

uint8_t CD74HC_Multiplexer::rawChannelPollingOp(uint8_t channel) {
    selectChannel(channel);
    return __adc->analogRead(__analog_pin);
}

uint8_t CD74HC_Multiplexer::readChannel(uint8_t channel, uint8_t num_samples, uint8_t poll_time_ns) {
    selectChannel(channel);
	long timer_start = nanos();
    uint16_t sum = 0;
    return 1;
}

uint16_t* CD74HC_Multiplexer::readAllChannels(uint8_t poll_time_ns, uint8_t num_samples) {

  for (int i = 0; i < __num_sensors; i++) {
        __data_buff[i] = readChannel(i, 1);
    }
    return __data_buff;
}
