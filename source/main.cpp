#include "lasm.hpp"
#include "plcm.hpp"

int main(int argc, const char * argv[]){

    int quitFlag = 0;
    extern double totalTime;
    extern int totalFrames;

    system("clear");

    while(quitFlag == 0){

        printf("-------------------------------------------------------------------------------\n");
        printf("\033[1mdemo for real-time chaotic video encryption based on multithreaded parallel\033[m\n");
        printf("\033[1mconfusion and diffusion\033[m\n\n");
        printf("input 0: exit.\n");
        printf("input 1: real-time encryption (confusion and diffusion) using PLCM.\n");
        printf("input 2: encryption and decryption using PLCM (there may exist some delay).\n");
        printf("input 3: real-time encryption (confusion and diffusion) using LASM.\n");
        printf("input 4: encryption and decryption using LASM (there may exist some delay).\n");
        printf("-------------------------------------------------------------------------------\n");
        printf("\033[1minput instruction: \033[m");

        int ins;
        scanf("%d", &ins);
        switch(ins){
            case 0: quitFlag = 1;
                    break;

            case 1: cplcmMainThreadDemo1();
                    printf("\033[1mAverage encryption time(CPLCM): %.2fms\033[m\n", (totalTime * 1000) / (double)totalFrames);
                    break;

            case 2: cplcmMainThreadDemo2();
                    break;            
            
            case 3: lasmMainThreadDemo3();
                    printf("\033[1mAverage encryption time(2DLASM): %.2fms\033[m\n", (totalTime * 1000) / (double)totalFrames);
                    break;

            case 4: lasmMainThreadDemo4();
                    break;

            default: system("clear");
                     printf("\033[1;31millegal instruction, try again\033[m\n");
                     break;
        }
    }

    return 0;
}

