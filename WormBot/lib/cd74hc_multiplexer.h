//
// Created by nikesh on 2/17/25.
//

#ifndef CD74HC_MULTIPLEXER_H
#define CD74HC_MULTIPLEXER_H
#include <Arduino.h>
#include <../include/ADC/ADC.h>


class CD74HC_Multiplexer {
public:
    CD74HC_Multiplexer();
    CD74HC_Multiplexer(uint8_t mux0, uint8_t mux1, uint8_t mux2, uint8_t mux3,
                       uint8_t analog_pin, uint8_t num_sensors,
                       ADC *adc);

    void selectChannel(uint8_t channel);


    uint8_t readChannel(uint8_t channel, uint8_t num_samples, uint8_t poll_time_ns);

    // read all channels
    // poll_time_ns is the time in nanoseconds to wait between reading each channel to get a sample
    // The Teensy is capable of 1M samples per second with two inbuilt ADCs
    uint16_t* readAllChannels(uint8_t poll_time_ns, uint8_t num_samples);



	/***
	@brief Polls the raw value of a channel without collecting multiple samples
	*/


	uint8_t rawChannelPollingOp(uint8_t channel);


private:
    uint16_t __mux0;
    uint16_t __mux1;
    uint16_t __mux2;
    uint16_t __mux3;
    uint8_t __analog_pin;
    uint8_t __num_sensors;
    uint16_t* __data_buff;
    ADC *__adc;
};

#endif //CD74HC_MULTIPLEXER_H
