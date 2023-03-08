#ifndef __LASM_HPP__
#define __LASM_HPP__

#include "plcm.hpp"

//the main thread call this function to present demo 3
int lasmMainThreadDemo3();

//the assistant threads call this function to present demo 3
static void * lasmAssistantThreadDemo3(void * arg);

//the main thread call this function to present demo 4
int lasmMainThreadDemo4();

//the assistant threads call this function to present demo 4
static void * lasmAssistantThreadDemo4(void * arg);

//iterate 2DLASM and return result
void LASM(double u, double x, double y, double * resultx, double * resulty);

//iterate 2DLASM and store results
void iterateLASM(double u, double x, double y, double * resultArray, int iterations);

//generate parameters for initializing 2DLASM
void generateParametersForLASM(double u, double * x, double * y, double * initParameterArray);

//generate confusion seed for confusion operations
int generateConfusionSeedForLASM(double u, double * x, double * y);

#endif
