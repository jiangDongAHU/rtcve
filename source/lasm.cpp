#include "lasm.hpp"

//global variables
//semaphore for synchronization and mutex
extern sem_t frameIsPreparedMutex[NUMBER_OF_THREADS], frameIsProcessedMutex[NUMBER_OF_THREADS];
extern Mat plainFrame, confusedFrame, encryptedFrame, tempFrame, decryptedFrame;
extern int frameWidth, frameHeight, videoFPS, totalFrames, confusionSeed;
extern double totalTime;


//the main thread call this function to present demo 3
int lasmMainThreadDemo3(){
    //open the video file
    VideoCapture capture;
    capture.open(VIDEO_FILE_NAME);
    if(!capture.isOpened()){
        printf("failed to open the video file!\n");
        return -1;
    }

    frameWidth = (int)capture.get(CAP_PROP_FRAME_WIDTH);
    frameHeight = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
    videoFPS = (int)capture.get(CAP_PROP_FPS);
    totalFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);

    //randomly select contro parameter and initial condition to initialize the mian PRBG
    srand(time(NULL));
    double x = (double) rand() / RAND_MAX;
    double y = (double) rand() / RAND_MAX;
    double u = (double) rand() / RAND_MAX;
    while(u < 0.37 || (u > 0.38 && u < 0.4) || (u > 0.42 && u < 0.44) || u > 0.93)
        u = (double) rand() / RAND_MAX;

    //the main threads generate a set of parameters to initialize the PRBGs of the assistant threads
    double * initParameterArray = (double *)malloc(6 * NUMBER_OF_THREADS * sizeof(double));
    int iterations = (frameWidth * frameHeight * 3 * CONFUSION_DIFFUSION_ROUNDS) / (BYTES_RESERVED * 2 * NUMBER_OF_THREADS);
    generateParametersForLASM(u, &x, &y, initParameterArray);

    //initialize the semaphores
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        sem_init(&frameIsPreparedMutex[i], 0, 0);
        sem_init(&frameIsProcessedMutex[i], 0, 0);
    }

    //create the assistant threads
    struct assistantThreadParameter tp[NUMBER_OF_THREADS];
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        tp[i].threadIdx = i;
        tp[i].iterations = iterations;
        tp[i].initParameterArray = initParameterArray;
    }
    
    pthread_t th[NUMBER_OF_THREADS];
    for(int i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_create(&th[i], NULL, lasmAssistantThreadDemo3, (void *)&tp[i]);

    totalTime = 0;

    int frameCount = 0;
    //fetch frame from video file
    while(capture.read(plainFrame)){
        double startTime = getCPUSecond();

        confusedFrame = plainFrame.clone();
        tempFrame = plainFrame.clone();
        encryptedFrame = plainFrame.clone();

        //generate confusion seed
        confusionSeed = abs(generateConfusionSeedForLASM(u, &x, &y)) % (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + CONFUSION_SEED_LOWWER_BOUND;

        //confusion and diffusion operation
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //plain frame is prepared, awake all assistant threads to perform confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = confusedFrame.clone();

            //diffusion step
            //awake all assistant threads to perform diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = encryptedFrame.clone();
        }


        imshow("plain video", plainFrame);
        imshow("encryption (confusion and diffusion) result", encryptedFrame);

        double endTime = getCPUSecond();

        totalTime = totalTime + (endTime- startTime);

        int waitKeyTime = 1000 / videoFPS - (int)((endTime - startTime) * 1000);
        if(waitKeyTime <= 0)
            waitKeyTime = 1;
        waitKey(waitKeyTime);
        
        if(frameCount % 10 == 0){
            system("clear");
            printf("\033[1mDemo 3: real-time encryption (confusion and diffusion) using 2DLASM.\033[m\n");
            printf("frame width: %d     | frame height: %d         | FPS: %d             | frames: %d\n", frameWidth, frameHeight, videoFPS, totalFrames);
            printf("assistant threads: %d | confusion rounds: %d       | diffusion rounds: %d | confusion seed: %d \n", NUMBER_OF_THREADS, CONFUSION_DIFFUSION_ROUNDS, CONFUSION_DIFFUSION_ROUNDS, confusionSeed);
            printf("1000/FPS: %dms       | encryption time: %.2fms  | frame index: %d\n", (int)(1000 / videoFPS), (endTime - startTime) * 1000, frameCount);
        }
        frameCount ++;

    }

    capture.release();
    free(initParameterArray);
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        sem_destroy(&frameIsPreparedMutex[i]);
        sem_destroy(&frameIsProcessedMutex[i]);
    }
    for(int i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_cancel(th[i]);
    destroyAllWindows();
    system("clear");

    return 0;
}

//the assistant threads call this function to present demo 3
static void * lasmAssistantThreadDemo3(void * arg){
    struct assistantThreadParameter * p = (struct assistantThreadParameter *)arg;
    int threadIdx = p->threadIdx;
    int iterations = p->iterations;
    double * initParameterArray = p->initParameterArray;
    int nextThreadIdx = (threadIdx + 1) % NUMBER_OF_THREADS;
    unsigned char diffusionSeed[3];

    int cols = frameWidth;
    int rows = frameHeight / NUMBER_OF_THREADS;
    int startRow = threadIdx * rows;
    int endRow = startRow + rows;

    double * resultArray1 = (double *)malloc(iterations * 2 * sizeof(double));
    double * resultArray2 = (double *)malloc(iterations * 2 * sizeof(double));
    unsigned char * uCharResultArray1 = (unsigned char *)malloc(iterations * 2 * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * uCharResultArray2 = (unsigned char *)malloc(iterations * 2 * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * byteSequence = (unsigned char *)malloc(iterations * 2 * BYTES_RESERVED * sizeof(unsigned char));

    //initialize 2DLASM
    double x1 = initParameterArray[threadIdx * 6];
    double y1 = initParameterArray[threadIdx * 6 + 1];
    double u1 = initParameterArray[threadIdx * 6 + 2];
    double x2 = initParameterArray[threadIdx * 6 + 3];
    double y2 = initParameterArray[threadIdx * 6 + 4];
    double u2 = initParameterArray[threadIdx * 6 + 5];

    //preiterate LASM
    double resultx, resulty;
    for(int i = 0; i < PRE_ITERATIONS; i++){
        LASM(u1, x1, y1, &resultx, & resulty);
        x1 = resultx;
        y1 = resulty;

        LASM(u2, x2, y2, &resultx, & resulty);
        x2 = resultx;
        y2 = resulty;
    }

    while(true){

        //generate byte sequence
        iterateLASM(u1, x1, y1, resultArray1, iterations);
        x1 = resultArray1[iterations * 2 - 2];
        y1 = resultArray1[iterations * 2 - 1];
        //convert iteration results into bytes
        convertResultToByte(resultArray1, uCharResultArray1, iterations * 2);
        iterateLASM(u2, x2, y2, resultArray2, iterations);
        x2 = resultArray2[iterations * 2 - 2];
        y2 = resultArray2[iterations * 2 - 1];
        //convert iteration results into byte sequence
        convertResultToByte(resultArray2, uCharResultArray2, iterations * 2);

        generateBytes(2 * iterations, uCharResultArray1, uCharResultArray2, byteSequence);

        int idx = 0;

        //performe confusion operation
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //wait the main thread to fetch a palin frame
            sem_wait(&frameIsPreparedMutex[threadIdx]);

            confusion(startRow, endRow);

            //complete a round of confusion
            sem_post(&frameIsProcessedMutex[threadIdx]);

            //performe diffusion operation
            sem_wait(&frameIsPreparedMutex[threadIdx]);

            //fetch the diffusion seeds
            diffusionSeed[0] = encryptedFrame.at<Vec3b>(rows * (nextThreadIdx + 1) - 1, frameWidth - 1)[0];
            diffusionSeed[1] = encryptedFrame.at<Vec3b>(rows * (nextThreadIdx + 1) - 1, frameWidth - 1)[1];
            diffusionSeed[2] = encryptedFrame.at<Vec3b>(rows * (nextThreadIdx + 1) - 1, frameWidth - 1)[2];

            idx = diffusion(startRow, endRow, diffusionSeed, byteSequence, idx);

            //complete a round of diffusion
            sem_post(&frameIsProcessedMutex[threadIdx]);
        }
    }

    free(resultArray1);
    free(resultArray2);
    free(uCharResultArray1);
    free(uCharResultArray2);
    free(byteSequence);

    return NULL;
}

//the main thread call this function to present demo 4
int lasmMainThreadDemo4(){
    //open the video file
    VideoCapture capture;
    capture.open(VIDEO_FILE_NAME);
    if(!capture.isOpened()){
        printf("failed to open the video file!\n");
        return -1;
    }

    frameWidth = (int)capture.get(CAP_PROP_FRAME_WIDTH);
    frameHeight = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
    videoFPS = (int)capture.get(CAP_PROP_FPS);
    totalFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);

    //randomly select contro parameter and initial condition to initialize the mian PRBG
    srand(time(NULL));
    double x = (double) rand() / RAND_MAX;
    double y = (double) rand() / RAND_MAX;
    double u = (double) rand() / RAND_MAX;
    while(u < 0.37 || (u > 0.38 && u < 0.4) || (u > 0.42 && u < 0.44) || u > 0.93)
        u = (double) rand() / RAND_MAX;

    //the main threads generate a set of parameters to initialize the PRBGs of the assistant threads
    double * initParameterArray = (double *)malloc(6 * NUMBER_OF_THREADS * sizeof(double));
    int iterations = (frameWidth * frameHeight * 3 * CONFUSION_DIFFUSION_ROUNDS) / (BYTES_RESERVED * 2 * NUMBER_OF_THREADS);
    generateParametersForLASM(u, &x, &y, initParameterArray);

    //initialize the semaphores
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        sem_init(&frameIsPreparedMutex[i], 0, 0);
        sem_init(&frameIsProcessedMutex[i], 0, 0);
    }

    //create the assistant threads
    struct assistantThreadParameter tp[NUMBER_OF_THREADS];
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        tp[i].threadIdx = i;
        tp[i].iterations = iterations;
        tp[i].initParameterArray = initParameterArray;
    }
    
    pthread_t th[NUMBER_OF_THREADS];
    for(int i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_create(&th[i], NULL, lasmAssistantThreadDemo4, (void *)&tp[i]);

    int frameCount = 0;
    //fetch frame from video file
    while(capture.read(plainFrame)){
        double startTime = getCPUSecond();

        confusedFrame = plainFrame.clone();
        tempFrame = plainFrame.clone();
        encryptedFrame = plainFrame.clone();

        //generate confusion seed
        confusionSeed = abs(generateConfusionSeedForLASM(u, &x, &y)) % (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + CONFUSION_SEED_LOWWER_BOUND;

        //encrypt the video frame
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //plain frame is prepared, awake all assistant threads to perform confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = confusedFrame.clone();

            //diffusion step
            //awake all assistant threads to perform diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = encryptedFrame.clone();
        }

        decryptedFrame = encryptedFrame.clone();

        //decrypt the encrypted frame
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //awake all assistant threads to perform inverse diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of inverse diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = decryptedFrame.clone();

            //inverse confusion step
            //awake all assistant threads to perform inverse confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of inverse confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = decryptedFrame.clone();
        }

        imshow("encryption (confusion and diffusion) result", encryptedFrame);
        imshow("decryption (inverse diffusion and confusion) result ", decryptedFrame);
        
        double endTime = getCPUSecond();

        int waitKeyTime = 1000 / videoFPS - (int)((endTime - startTime) * 1000);
        if(waitKeyTime <= 0)
            waitKeyTime = 1;
        waitKey(waitKeyTime);
        
        if(frameCount % 10 == 0){
            system("clear");
            printf("\033[1mDemo 4: encryption and decryption using 2DLASM (there may exist some delay).\033[m\n");
            printf("frame width: %d     | frame height: %d         | FPS: %d             | frames: %d\n", frameWidth, frameHeight, videoFPS, totalFrames);
            printf("assistant threads: %d | confusion rounds: %d       | diffusion rounds: %d | confusion seed: %d \n", NUMBER_OF_THREADS, CONFUSION_DIFFUSION_ROUNDS, CONFUSION_DIFFUSION_ROUNDS, confusionSeed);
            printf("1000/FPS: %dms       | total time: %.2fms       | frame index: %d\n", (int)(1000 / videoFPS), (endTime - startTime) * 1000, frameCount);
        }
        frameCount ++;

    }

    capture.release();
    free(initParameterArray);
    for(int i = 0; i < NUMBER_OF_THREADS; i++){
        sem_destroy(&frameIsPreparedMutex[i]);
        sem_destroy(&frameIsProcessedMutex[i]);
    }
    for(int i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_cancel(th[i]);
    destroyAllWindows();
    system("clear");

    return 0;
}

//the assistant threads call this function to present demo 4
static void * lasmAssistantThreadDemo4(void * arg){
    struct assistantThreadParameter * p = (struct assistantThreadParameter *)arg;
    int threadIdx = p->threadIdx;
    int iterations = p->iterations;
    double * initParameterArray = p->initParameterArray;
    int nextThreadIdx = (threadIdx + 1) % NUMBER_OF_THREADS;
    unsigned char diffusionSeed[3];
    unsigned char diffusionSeedArray[3 * CONFUSION_DIFFUSION_ROUNDS];

    int cols = frameWidth;
    int rows = frameHeight / NUMBER_OF_THREADS;
    int startRow = threadIdx * rows;
    int endRow = startRow + rows;

    double * resultArray = (double *)malloc(iterations * 2 * sizeof(double));
    double * resultArray1 = (double *)malloc(iterations * 2 * sizeof(double));
    double * resultArray2 = (double *)malloc(iterations * 2 * sizeof(double));
    unsigned char * uCharResultArray1 = (unsigned char *)malloc(iterations * 2 * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * uCharResultArray2 = (unsigned char *)malloc(iterations * 2 * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * byteSequence = (unsigned char *)malloc(iterations * 2 * BYTES_RESERVED * sizeof(unsigned char));

    //initialize 2DLASM
    double x1 = initParameterArray[threadIdx * 6];
    double y1 = initParameterArray[threadIdx * 6 + 1];
    double u1 = initParameterArray[threadIdx * 6 + 2];
    double x2 = initParameterArray[threadIdx * 6 + 3];
    double y2 = initParameterArray[threadIdx * 6 + 4];
    double u2 = initParameterArray[threadIdx * 6 + 5];

    //preiterate LASM
    double resultx, resulty;
    for(int i = 0; i < PRE_ITERATIONS; i++){
        LASM(u1, x1, y1, &resultx, & resulty);
        x1 = resultx;
        y1 = resulty;

        LASM(u2, x2, y2, &resultx, & resulty);
        x2 = resultx;
        y2 = resulty;
    }

    while(true){

        //generate byte sequence
        iterateLASM(u1, x1, y1, resultArray1, iterations);
        x1 = resultArray1[iterations * 2 - 2];
        y1 = resultArray1[iterations * 2 - 1];
        convertResultToByte(resultArray1, uCharResultArray1, iterations * 2);
        iterateLASM(u2, x2, y2, resultArray2, iterations);
        x2 = resultArray2[iterations * 2 - 2];
        y2 = resultArray2[iterations * 2 - 1];
        convertResultToByte(resultArray2, uCharResultArray2, iterations * 2);
        generateBytes(2 * iterations, uCharResultArray1, uCharResultArray2, byteSequence);

        int idx = 0;
        int diffusionSeedArrayIndex = 0;  
        //encrypt the video frame
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //wait the main thread to fetch a palin frame
            sem_wait(&frameIsPreparedMutex[threadIdx]);

            confusion(startRow, endRow);

            //complete a round of confusion
            sem_post(&frameIsProcessedMutex[threadIdx]);       

            //perform diffusion operation
            sem_wait(&frameIsPreparedMutex[threadIdx]);

            //fetch the diffusion seeds
            diffusionSeed[0] = encryptedFrame.at<Vec3b>(rows * (nextThreadIdx + 1) - 1, frameWidth - 1)[0];
            diffusionSeed[1] = encryptedFrame.at<Vec3b>(rows * (nextThreadIdx + 1) - 1, frameWidth - 1)[1];
            diffusionSeed[2] = encryptedFrame.at<Vec3b>(rows * (nextThreadIdx + 1) - 1, frameWidth - 1)[2];
            diffusionSeedArray[diffusionSeedArrayIndex ++] = diffusionSeed[0];
            diffusionSeedArray[diffusionSeedArrayIndex ++] = diffusionSeed[1];
            diffusionSeedArray[diffusionSeedArrayIndex ++] = diffusionSeed[2];

            idx = diffusion(startRow, endRow, diffusionSeed, byteSequence, idx);

            //complete a round of diffusion
            sem_post(&frameIsProcessedMutex[threadIdx]);
        }

        //decrypt the encrypted video frame
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //wait the main thread to awake the assistant threads
            sem_wait(&frameIsPreparedMutex[threadIdx]);

            //fetch the diffusion seeds
            diffusionSeed[2] = diffusionSeedArray[--diffusionSeedArrayIndex];
            diffusionSeed[1] = diffusionSeedArray[--diffusionSeedArrayIndex];
            diffusionSeed[0] = diffusionSeedArray[--diffusionSeedArrayIndex];

            idx = inverseDiffusion(startRow, endRow, diffusionSeed, byteSequence, idx);

            //complete a round of inverse diffusion operation
            sem_post(&frameIsProcessedMutex[threadIdx]);

            //perform inverse confusion operation
            sem_wait(&frameIsPreparedMutex[threadIdx]);

            inverseConfusion(startRow, endRow);

            //complete a round of inverse confusion operation
            sem_post(&frameIsProcessedMutex[threadIdx]);
        }
    }

    free(resultArray1);
    free(resultArray2);
    free(uCharResultArray1);
    free(uCharResultArray2);
    free(byteSequence);

    return NULL;
}

//iterate 2DLASM iterations times and store results in array
void iterateLASM(double u, double x, double y, double * resultArray, int iterations){
    double resultx, resulty;

    int idx = 0;
    //Iterate 2DLASM, generate results, and store results in array 
    for(int i = 0; i < iterations; i++){
        LASM(u, x, y, &resultx, &resulty);
        x = resultx;
        y = resulty;
        resultArray[idx++] = x;
        resultArray[idx++] = y;
    }
}

//generate parameters for initializing 2DLASM
void generateParametersForLASM(double u, double * x, double * y, double * initParameterArray){
    //preiterate the 2DLASM
    double resultx, resulty;
    for(int i = 0; i < PRE_ITERATIONS; i++){
        LASM(u, * x, * y, &resultx, &resulty);
        * x = resultx;
        * y = resulty;
    }

    //iterate the main PRBG and generate enouth parameters for initializing the assitant threads
    int idx = 0;
    while(idx < NUMBER_OF_THREADS * 6){
        LASM(u, * x, * y, &resultx, & resulty);
        //generate x and y
        * x = resultx;
        * y = resulty;
        initParameterArray[idx++] = * x;
        initParameterArray[idx++] = * y;

        //generate u
        LASM(u, * x, * y, &resultx, &resulty);
        * x = resultx;
        * y = resulty;
        while((* x < 0.37 || (* x > 0.38 && * x < 0.4) || (* x > 0.42 && * x < 0.44) || * x > 0.93) && 
              (* y < 0.37 || (* y > 0.38 && * y < 0.4) || (* y > 0.42 && * y < 0.44) || * y > 0.93)){
            LASM(u, * x, * y, &resultx, & resulty);
            * x = resultx;
            * y = resulty;
            }
            if((* x >= 0.37 && * x <= 0.38) || (* x >= 0.4 && * x <= 0.42) || (* x >= 0.44 && * x <= 0.93)){
                initParameterArray[idx++] = * x;
            }
            else if ((* y >= 0.37 && * y <= 0.38) || (* y >= 0.4 && * y <= 0.42) || (* y >= 0.44 && * y <= 0.93))
            {
                initParameterArray[idx++] = * y;
            }   
    }
}

//generate confusion seed for confusion operations
int generateConfusionSeedForLASM(double u, double * x, double * y){
    double resultx, resulty;
    LASM(u, * x, * y, &resultx, &resulty);
    * x = resultx;
    * y = resulty;
    
    int confusionSeedResult = 0;
    memcpy(&confusionSeedResult, (unsigned char *)&resultx, BYTES_RESERVED);
    return confusionSeedResult;
}
