#ifndef __CPLCM_HPP__
#define __CPLCM_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

using namespace cv;

#define VIDEO_FILE_NAME "video/plainVideo.mp4"
#define NUMBER_OF_THREADS 8
#define CONFUSION_DIFFUSION_ROUNDS 5

#define CONFUSION_SEED_UPPER_BOUND 10000
#define CONFUSION_SEED_LOWWER_BOUND 3000
#define PRE_ITERATIONS 1000
#define BYTES_RESERVED 6
#define PI acos(-1)

//parameters for creating the assistant threads
struct assistantThreadParameter{
    int threadIdx;
    int iterations;
    double * initParameterArray;
};

//the main thread call this function to present demo 1
int cplcmMainThreadDemo1();

//the assistant threads call this function to present demo 1
static void * cplcmAssistantThreadDemo1(void * arg);

//the main thread call this function to present demo 2
int cplcmMainThreadDemo2();

//the assistant threads call this function to present demo 2
static void * cplcmAssistantThreadDemo2(void * arg);

//Iterate PLCM and return iteration result
double PLCM(double controlParameter, double initialCondition);

//Iterate a PLCM and store results
double iteratePLCM(double controlParameter, double initialCondition, int iterations, double * iterationResultArray);

//Act XOR operation and store the bytes for encryption
void generateBytes(int iterations, unsigned char * uCharResultArray1, unsigned char * uCharResultArray2, unsigned char * byteSequence);

//generate parameters for initializing assistant threads' PRBGs
void generateParametersPLCM(double controlParameter1, double * initialCondition1, 
                            double controlParameter2, double * initialCondition2,
                            double * initParameterArray);

//generate confusion seed for confusion operations
int generateConfusionSeedPLCM(double controlParameter, double * initialCondition);

//generate diffusion seedd for diffusion operations
void generateDiffusionSeedPLCM(double controlParameter, double * initialCondition, unsigned char * diffusionSeedArray);

//confusion function
void confusion(int startRow, int endRow);

//inverse confusion function
void inverseConfusion(int startRow, int endRow);

//diffusion function
int diffusion(int startRow, int endRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx);

//inverse diffusion function
int inverseDiffusion(int startRow, int endRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx);

//convert iteration results to byte sequence
void convertResultToByte(double * resultArray, unsigned char * byteSequence, int elems);

//get cpu time
double getCPUSecond(void);

#endif
