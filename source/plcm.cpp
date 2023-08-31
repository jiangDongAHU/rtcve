#include "plcm.hpp"

//global variables
//semaphore for synchronization and mutex
sem_t frameIsPreparedMutex[NUMBER_OF_THREADS], frameIsProcessedMutex[NUMBER_OF_THREADS];
Mat plainFrame, confusedFrame, encryptedFrame, tempFrame, decryptedFrame;
int frameWidth, frameHeight, videoFPS, totalFrames, confusionSeed;
double totalTime;

//the main thread call this function to present demo 1
int cplcmMainThreadDemo1(){
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

    //randomly select initial condition and control parameter
    srand(time(NULL));
    //randomly select initial conditions and control parameters
    srand(time(NULL));
    double controlParameter1 = (double) rand() / RAND_MAX;
    if(controlParameter1 > 0.5)
        controlParameter1 = 1 - controlParameter1;
    double initialCondition1 = (double) rand() / RAND_MAX;

    double controlParameter2 = (double) rand() / RAND_MAX;
    if(controlParameter2 > 0.5)
        controlParameter2 = 1 - controlParameter2;
    double initialCondition2 = (double) rand() / RAND_MAX;

   
    //generate a set of parameters for initializing the PRBGs of the assistant threads
    double * initParameterArray = (double *)malloc(4 * NUMBER_OF_THREADS * sizeof(double));
    int iterations = (frameWidth * frameHeight * 3 * CONFUSION_DIFFUSION_ROUNDS) / (BYTES_RESERVED * NUMBER_OF_THREADS);
    generateParametersPLCM(controlParameter1, &initialCondition1, controlParameter2, &initialCondition2, initParameterArray);

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
        pthread_create(&th[i], NULL, cplcmAssistantThreadDemo1, (void *)&tp[i]);

    totalTime = 0;

    int frameCount = 0;
    //fetch and encrypt frames from video file
    while(capture.read(plainFrame)){
        double startTime = getCPUSecond();

        confusedFrame = plainFrame.clone();
        tempFrame = plainFrame.clone();
        encryptedFrame = plainFrame.clone();

        //generate confusion seed
        confusionSeed  = abs(generateConfusionSeedPLCM(controlParameter1, &initialCondition1)) % 
                         (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + 
                         CONFUSION_SEED_LOWWER_BOUND;

        //confusion and diffusion operation
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //confusion operation
            //plain frame is prepared, awake all assistant threads to perform confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = confusedFrame.clone();

            //diffusion operation
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

        //char encryptedFramePath[100];
        //sprintf(encryptedFramePath, "images/encryptedFrame%d.tif", frameCount);
        //imwrite(encryptedFramePath, encryptedFrame);

        double endTime = getCPUSecond();
        totalTime = totalTime + (endTime - startTime);

        int waitKeyTime = 1000 / videoFPS - (int)((endTime - startTime) * 1000);
        if(waitKeyTime <= 0)
            waitKeyTime = 1;
        waitKey(waitKeyTime);
        
        if(frameCount % 10 == 0){
            system("clear");
            printf("\033[1mDemo 1: real-time encryption (confusion and diffusion) using PLCM.\033[m\n");
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

//the assistant threads call this function to present demo 1
static void * cplcmAssistantThreadDemo1(void * arg){
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

    double * iterationResultArray1 = (double *)malloc(iterations * sizeof(double));
    double * iterationResultArray2 = (double *)malloc(iterations * sizeof(double));
    unsigned char * uCharResultArray1 = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * uCharResultArray2 = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * byteSequence = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));

    //initialize PLCMs
    double controlParameter1 = initParameterArray[threadIdx * 4];
    if(controlParameter1 > 0.5)
        controlParameter1 = 1 - controlParameter1;
    double initialCondition1 = initParameterArray[threadIdx * 4 + 1];

    double controlParameter2 = initParameterArray[threadIdx * 4 + 2];
    if(controlParameter2 > 0.5)
        controlParameter2 = 1 - controlParameter2;
    double initialCondition2 = initParameterArray[threadIdx * 4 + 3];

    //preiterate PLCMs
    for(int i = 0; i < PRE_ITERATIONS; i++){
        initialCondition1 = PLCM(controlParameter1, initialCondition1);
        initialCondition2 = PLCM(controlParameter2, initialCondition2);
    }

    while(true){
        //generate byte sequence
        initialCondition1 = iteratePLCM(controlParameter1, initialCondition1, iterations, iterationResultArray1);
        initialCondition2 = iteratePLCM(controlParameter2, initialCondition2, iterations, iterationResultArray2);
        convertResultToByte(iterationResultArray1, uCharResultArray1, iterations);
        convertResultToByte(iterationResultArray2, uCharResultArray2, iterations);
        generateBytes(iterations, uCharResultArray1, uCharResultArray2, byteSequence);

        int idx = 0;
        
        //perform confusion and diffusion operations
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

    free(iterationResultArray1);
    free(iterationResultArray2);
    free(uCharResultArray1);  
    free(uCharResultArray2);
    free(byteSequence);

    return NULL;
}

//the main thread call this function to present demo 2
int cplcmMainThreadDemo2(){
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

    //randomly select initial condition and control parameter
    srand(time(NULL));
    double controlParameter1 = (double) rand() / RAND_MAX;
    if(controlParameter1 > 0.5)
        controlParameter1 = 1 - controlParameter1;
    double initialCondition1 = (double) rand() / RAND_MAX;

    double controlParameter2 = (double) rand() / RAND_MAX;
    if(controlParameter2 > 0.5)
        controlParameter2 = 1 - controlParameter2;
    double initialCondition2 = (double) rand() / RAND_MAX;

    //generate a set of parameters for initializing the PRBGs of the assistant threads
    double * initParameterArray = (double *)malloc(4 * NUMBER_OF_THREADS * sizeof(double));
    int iterations = (frameWidth * frameHeight * 3 * CONFUSION_DIFFUSION_ROUNDS) / (BYTES_RESERVED * NUMBER_OF_THREADS);
    generateParametersPLCM(controlParameter1, &initialCondition1, controlParameter2, &initialCondition2, initParameterArray);

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
        pthread_create(&th[i], NULL, cplcmAssistantThreadDemo2, (void *)&tp[i]);

    int frameCount = 0;
    //fetch frame from video file
    while(capture.read(plainFrame)){
        double startTime = getCPUSecond();

        confusedFrame = plainFrame.clone();
        tempFrame = plainFrame.clone();
        encryptedFrame = plainFrame.clone();

        //generate confusion seed
        confusionSeed  = abs(generateConfusionSeedPLCM(controlParameter1, &initialCondition1)) % 
                         (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + 
                         CONFUSION_SEED_LOWWER_BOUND;

        //perform confusion and diffusion operations
        for(int i = 0; i < CONFUSION_DIFFUSION_ROUNDS; i++){
            //plain frame is prepared, awake all assistant threads to perform confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads to complete a round of confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = confusedFrame.clone();

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

            //wait all assistant threads complete a round of inverse confusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_wait(&frameIsProcessedMutex[j]);

            tempFrame = decryptedFrame.clone();

            //awake all assistant threads to perform inverse diffusion operation
            for(int j = 0; j < NUMBER_OF_THREADS; j++)
                sem_post(&frameIsPreparedMutex[j]);

            //wait all assistant threads complete a round of inverse diffusion operation
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
            printf("\033[1mDemo 2: encryption and decryption using PLCM (there may exist some delay).\033[m\n");
            printf("frame width: %d     | frame height: %d         | FPS: %d             | frames: %d\n", frameWidth, frameHeight, videoFPS, totalFrames);
            printf("assistant threads: %d | confusion rounds: %d       | diffusion rounds: %d | confusion seed: %d \n", NUMBER_OF_THREADS, CONFUSION_DIFFUSION_ROUNDS, CONFUSION_DIFFUSION_ROUNDS, confusionSeed);
            printf("1000/FPS: %dms       | total time: %.2fms  | frame index: %d\n", (int)(1000 / videoFPS), (endTime - startTime) * 1000, frameCount);
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

//the assistant threads call this function to present demo 2
static void * cplcmAssistantThreadDemo2(void * arg){
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

    double * iterationResultArray1 = (double *)malloc(iterations * sizeof(double));
    double * iterationResultArray2 = (double *)malloc(iterations * sizeof(double));
    unsigned char * uCharResultArray1 = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * uCharResultArray2 = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * byteSequence = (unsigned char *)malloc(iterations * BYTES_RESERVED * sizeof(unsigned char));

    //initialize PLCMs
    double controlParameter1 = initParameterArray[threadIdx * 4];
    if(controlParameter1 > 0.5)
        controlParameter1 = 1 - controlParameter1;
    double initialCondition1 = initParameterArray[threadIdx * 4 + 1];

    double controlParameter2 = initParameterArray[threadIdx * 4 + 2];
    if(controlParameter2 > 0.5)
        controlParameter2 = 1 - controlParameter2;
    double initialCondition2 = initParameterArray[threadIdx * 4 + 3];

    //preiterate PLCMs
    for(int i = 0; i < PRE_ITERATIONS; i++){
        initialCondition1 = PLCM(controlParameter1, initialCondition1);
        initialCondition2 = PLCM(controlParameter2, initialCondition2);
    }

    while(true){
        //generate byte sequence
        initialCondition1 = iteratePLCM(controlParameter1, initialCondition1, iterations, iterationResultArray1);
        initialCondition2 = iteratePLCM(controlParameter2, initialCondition2, iterations, iterationResultArray2);
        convertResultToByte(iterationResultArray1, uCharResultArray1, iterations);
        convertResultToByte(iterationResultArray2, uCharResultArray2, iterations);
        generateBytes(iterations, uCharResultArray1, uCharResultArray2, byteSequence);

        int idx = 0;     
        int diffusionSeedArrayIndex = 0;   
        //perform confusion and diffusion operations
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
            diffusionSeedArray[diffusionSeedArrayIndex ++] = diffusionSeed[0];
            diffusionSeedArray[diffusionSeedArrayIndex ++] = diffusionSeed[1];
            diffusionSeedArray[diffusionSeedArrayIndex ++] = diffusionSeed[2];

            idx = diffusion(startRow, endRow, diffusionSeed, byteSequence, idx);

            //complete a round of diffusion
            sem_post(&frameIsProcessedMutex[threadIdx]);
        }

        //decrypt the encrypted frame
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

    free(iterationResultArray1);
    free(iterationResultArray2);
    free(uCharResultArray1);  
    free(uCharResultArray2);
    free(byteSequence);

    return NULL;
}

//iterate PLCM and return iteration result
double PLCM(double controlParameter, double initialCondition){
    double iterationResult = 0;

    if(initialCondition >= 0 && initialCondition <= controlParameter)
        iterationResult = initialCondition / controlParameter;
    
    else if(initialCondition > controlParameter && initialCondition <= 0.5)
        iterationResult = (initialCondition - controlParameter) / (0.5 - controlParameter);
    
    else
        iterationResult = PLCM(controlParameter, 1 - initialCondition);

    return iterationResult;
}

//iterate a PLCM and store results
double iteratePLCM(double controlParameter, double initialCondition, int iterations, double * iterationResultArray){
    double iterationResult = 0;

    for(int i = 0; i < iterations; i ++){
        iterationResult = PLCM(controlParameter, initialCondition);
        initialCondition = iterationResult;
        iterationResultArray[i] = iterationResult;
    }

    return initialCondition;
}

//Act XOR operation and store the bytes for encryption
void generateBytes(int iterations, unsigned char * uCharResultArray1, unsigned char * uCharResultArray2, unsigned char * byteSequence){
     int n = iterations * BYTES_RESERVED;
    
    for(int i = 0; i < n; i++ )
        byteSequence[i] = uCharResultArray1[i] ^ uCharResultArray2[i];
}

//generate parameters for initializing assistant threads' PRBGs
void generateParametersPLCM(double controlParameter1, double * initialCondition1, 
                            double controlParameter2, double * initialCondition2,
                            double * initParameterArray){
    for(int i = 0; i < PRE_ITERATIONS; i++){
        * initialCondition1 = PLCM(controlParameter1, * initialCondition1);
        * initialCondition2 = PLCM(controlParameter2, * initialCondition2);
    }

    for(int i = 0; i < NUMBER_OF_THREADS * 4; i++){
        if(i % 2 == 0){
            initParameterArray[i] = PLCM(controlParameter1, * initialCondition1);
            * initialCondition1   = initParameterArray[i];
        }

        else
            initParameterArray[i] = PLCM(controlParameter2, * initialCondition2);
            * initialCondition2   = initParameterArray[i];
    }
}

//generate confusion seed for confusion operations
int generateConfusionSeedPLCM(double controlParameter, double * initialCondition){
    * initialCondition      = PLCM(controlParameter, * initialCondition);
    double iterationResult  = * initialCondition;
    int confusionSeedResult = 0;

    memcpy(&confusionSeedResult, (unsigned char *)&iterationResult, BYTES_RESERVED);
    return confusionSeedResult;   
}

//generate diffusion seedd for diffusion operations
void generateDiffusionSeedPLCM(double controlParameter, double * initialCondition, unsigned char * diffusionSeedArray){
    for(int i = 0; i < 3 * CONFUSION_DIFFUSION_ROUNDS; i++){
        * initialCondition     = PLCM(controlParameter, * initialCondition);
        double iterationResult = * initialCondition;

        unsigned char * diffusionSeed = &diffusionSeedArray[i];
        memcpy(diffusionSeed, (unsigned char *)&iterationResult, 1);
    }
}

//confusion function
void confusion(int startRow, int endRow){
    for(int r = startRow; r < endRow; r++)
        for(int c = 0; c < frameWidth; c++){
            int nr = (r + c) % frameHeight;// + startRow;
            int temp = round(confusionSeed * sin(2 * PI * nr / frameHeight));
            int nc = ((c + temp) % frameWidth + frameWidth) % frameWidth;

            confusedFrame.at<Vec3b>(nr, nc)[0] = tempFrame.at<Vec3b>(r, c)[0];
            confusedFrame.at<Vec3b>(nr, nc)[1] = tempFrame.at<Vec3b>(r, c)[1];
            confusedFrame.at<Vec3b>(nr, nc)[2] = tempFrame.at<Vec3b>(r, c)[2];
        }
}


//inverse confusion function
void inverseConfusion(int startRow, int endRow){
    for(int r = startRow; r < endRow; r++)
        for(int c = 0; c < frameWidth; c++){
            int temp = round(confusionSeed * sin(2 * PI * r / frameHeight));
            int nr = ((r - c + temp)% frameHeight + frameHeight) % frameHeight;
            int nc = ((c - temp) % frameWidth + frameWidth) % frameWidth;

            decryptedFrame.at<Vec3b>(nr, nc)[0] = tempFrame.at<Vec3b>(r, c)[0];
            decryptedFrame.at<Vec3b>(nr, nc)[1] = tempFrame.at<Vec3b>(r, c)[1];
            decryptedFrame.at<Vec3b>(nr, nc)[2] = tempFrame.at<Vec3b>(r, c)[2];
        }
}


//iterate 2DLASM and return result
void LASM(double u, double x, double y, double * resultx, double * resulty){
    * resultx = sin(PI * u * (y + 3) * x * (1 - x));
    * resulty = sin(PI * u * (* resultx + 3) * y * (1 - y));
}

//diffusion function
int diffusion(int startRow, int endRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx){
    int prei, prej;

    for(int i = startRow ; i < endRow; i++)
        for(int j = 0; j < frameWidth; j++){
            if(j != 0){
                prei = i;
                prej = j - 1;
                encryptedFrame.at<Vec3b>(i, j)[0] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[0] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[0];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[1] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[1] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[1];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[2] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[2] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[2];
                idx = idx + 1;
            }

            else if(i != startRow && j == 0){
                prei = i - 1;
                prej = frameWidth - 1;
                encryptedFrame.at<Vec3b>(i, j)[0] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[0] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[0];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[1] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[1] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[1];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[2] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[2] + byteSequence[idx]) % 256) ^ encryptedFrame.at<Vec3b>(prei, prej)[2];
                idx = idx + 1;
            }

            else if(i == startRow && j == 0){
                encryptedFrame.at<Vec3b>(i, j)[0] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[0] + byteSequence[idx]) % 256) ^ diffusionSeed[0];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[1] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[1] + byteSequence[idx]) % 256) ^ diffusionSeed[1];
                idx = idx + 1;
                encryptedFrame.at<Vec3b>(i, j)[2] = byteSequence[idx] ^ ((tempFrame.at<Vec3b>(i,j)[2] + byteSequence[idx]) % 256) ^ diffusionSeed[2];
                idx = idx + 1;
            }
        }
    
    return idx;
}

//inverse diffusion function
int inverseDiffusion(int startRow, int endRow, unsigned char * diffusionSeed, unsigned char * byteSequence, int idx){
    int prei, prej;

    for(int i = endRow - 1; i >= startRow; i--)
        for(int j = frameWidth - 1; j >= 0; j--){
            if(j != 0){
                prei = i;
                prej = j - 1;

                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[2] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[2] ^ tempFrame.at<Vec3b>(prei, prej)[2]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[1] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[1] ^ tempFrame.at<Vec3b>(prei, prej)[1]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[0] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[0] ^ tempFrame.at<Vec3b>(prei, prej)[0]) + 256 - byteSequence[idx]);
            }

            else if(i != startRow && j == 0){
                prei = i - 1;
                prej = frameWidth - 1;

                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[2] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[2] ^ tempFrame.at<Vec3b>(prei, prej)[2]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[1] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[1] ^ tempFrame.at<Vec3b>(prei, prej)[1]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[0] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[0] ^ tempFrame.at<Vec3b>(prei, prej)[0]) + 256 - byteSequence[idx]);
            }

            else if(i == startRow && j == 0){
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[2] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[2] ^ diffusionSeed[2]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[1] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[1] ^ diffusionSeed[1]) + 256 - byteSequence[idx]);
                idx = idx - 1;
                decryptedFrame.at<Vec3b>(i, j)[0] = ((byteSequence[idx] ^ tempFrame.at<Vec3b>(i, j)[0] ^ diffusionSeed[0]) + 256 - byteSequence[idx]);
            }

        }

    return idx;
}

//convert iteration results to byte sequence
void convertResultToByte(double * resultArray, unsigned char * byteSequence, int elems){
    unsigned char * p;
    for(int i = 0; i < elems; i++){
        p = &byteSequence[i * BYTES_RESERVED];
        memcpy(p, (unsigned char *)&resultArray[i], BYTES_RESERVED);   
    }
}

//get cpu time
double getCPUSecond(void){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
