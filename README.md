## Real-Time Chaotic Video Encryption Based on Multithreaded Parallel Confusion and Diffusion

### Development Enviroment

* Operating System : Ubuntu 20.04
* OpenCV : 4.2.0
* Programming Language : C/C++
* Complier : g++ 9.4.0

### File Description

The original videos and source files are stored in separate directories: "video" for videos and "source" for source files. All original videos can be downloaded from the video trace library at: http://trace.eas.asu.edu/yuv/index.html

Before running the demo, make sure to install OpenCV, g++, ffmpeg, python3, and the tkinter module.

To start the demo, simply run the Python script "demo.py" and select the original video you want to use. The script will automatically convert the selected video from YUV420 (.yuv) format to RGB24 (.mp4) format at 24 frames per second (24FPS), compile the source files,  and run the demo.

The number of assistant threads, rounds of confusion and diffusion operations, etc., are declared in source/plcm.hpp:

```
#define VIDEO_FILE_NAME "video/plainVideo.mp4"

#define NUMBER_OF_THREADS 8
#define CONFUSION_DIFFUSION_ROUNDS 5

#define CONFUSION_SEED_UPPER_BOUND 10000
#define CONFUSION_SEED_LOWWER_BOUND 3000
#define PRE_ITERATIONS 1000
#define BYTES_RESERVED 6
#define PI acos(-1)
```

Recommended setting:

* the width of the frame == the height of the frame
* the width and the height of the frame % NUMBER_OF_THREADS == 0


### Demo Description
The main thread randomly selects a set of parameters to initialize its PRBG, and demonstrates different demos according to the input:

* Input 0 : exit the program
* Input 1 : real-time encryption using PLCM.
* input 2 : encryption and decryption using PLCM (there could be some delay).
* Input 3 : real-time encryption using LASM.
* input 4 : encryption and decryption using LASM (there could be some delay).

