//
// Created by nikesh on 2/17/25.
//
#prama once

#include <Arduino.h>
#include <../include/ADC/ADC.h>

ADC *adc0;
ADC *adc1;

inline ADC** config_adcs(uint8_t averaging, uint8_t resolution, ADC_CONVERSION_SPEED conversion_speed, ADC_SAMPLING_SPEED sampling_speed){

    adc0 = new ADC();
    adc1 = new ADC();

    adc0->setAveraging(averaging);
    adc0->setResolution(resolution);
    adc0->setConversionSpeed(conversion_speed);
    adc0->setSamplingSpeed(sampling_speed);

    adc1->setAveraging(averaging);
    adc1->setResolution(resolution);
    adc1->setConversionSpeed(conversion_speed);
    adc1->setSamplingSpeed(sampling_speed);

    ADC** adcs = new ADC*[2];
    adcs[0] = adc0;
    adcs[1] = adc1;

    return adcs;
}

