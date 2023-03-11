import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()

#select the original video
filePath = filedialog.askopenfilename()

#convert the original video from YUV420 into RGB24 format.
videoSize = 480
os.system('rm video/plainVideo.mp4')
os.system('ffmpeg -s 352x288 -pix_fmt yuv420p -i ' + filePath + ' -pix_fmt rgb24' + ' video/output.rgb')
os.system('ffmpeg -f rawvideo -pix_fmt rgb24 -s 352X288 -r 24 -i ' + 'video/output.rgb -vcodec libx265 -x265-params lossless=1 -pix_fmt yuv420p -r 24 -s ' + str(videoSize) + 'x' + str(videoSize) + ' video/plainVideo.mp4')
os.system('rm video/output.rgb')

#compile the source files and run the demo
print('\nCompiling the source files...')
os.system('g++ source/main.cpp source/plcm.cpp source/lasm.cpp -o demo -lpthread -lm `pkg-config opencv4 --libs --cflags`')
os.system('./demo')
os.system('rm video/plainVideo.mp4')
os.system('rm demo')
