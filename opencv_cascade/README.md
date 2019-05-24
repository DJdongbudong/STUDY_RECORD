# ubuntu 下进行的级联器训练
## 文件
root/

    neg/**.bmp
    
    pos/**.bmp
    
    opencv_createsamples
    
    opencv_traincascade
    
## 数据设置
### 批量改变尺寸
```
for x in *.jpg; do convert -resize 256x256\! $x $x; done
```
### 批量改变名字
```
rename "s/^/image_/" x.bmp
```
### 批量改变名字
```
i=0;for x in *; do mv $x $i.bmp; let i=i+1; done
```
### 批量写名字名到文件
```
ls ./neg/*.* >neg.txt
ls ./pos/*.* >pos.txt
```

### 修改 pos.txt >>> xx.bmp 1 x y w h <<<

## 数据类型转换——生成vec文件   
### 修改样本截取大小
sudo ./opencv_createsamples \
-info pos.txt \
-bg neg.txt \
-num 7092 \
-vec pos.vec \
-w 40 -h 60

## 数据训练 —— 重新训练时候保证model里面是空的
### 创建 model 文件夹、样本数量、尺寸、LBP 或者　HAAR、占用内存、准确率、错误率、使用的模型
sudo ./opencv_traincascade \
-data model \
-vec pos.vec -bg neg.txt \
-numPos 7000 -numNeg 100 \
-numStages 20 \
-w 40 -h 60 \
-featureType LBP \
-precalcValBufSize 12400 -precalcIdxBufSize 12400 \
-minHitRate 0.9999  -maxFalseAlarmRate 0.25 \
-mode ALL

## 模型读取检测
模型复制到文件夹detector
python objectdetection.py
python objectdetection.py --image = xx.jpg
python objectdetection.py --video = xx.mp4
``` objectdetection.py内容如下：
# -*- coding: utf-8 -*-
import cv2
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Process inputs
winName = 'Object detection in OpenCV-DJH'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
outputFile = "out.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'out.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4]+'out.avi'
else:
    # Webcam input
    cap = cv2.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv2.VideoWriter(outputFile, \
        cv2.VideoWriter_fourcc('M','J','P','G'), 30, \
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),\
            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))



if __name__ == '__main__':   
    ## >>>>> 其他 <<<<<< ##
    Cascade = cv2.CascadeClassifier(r'./model/cascade.xml') 
    
    ## >>>>> 人眼 <<<<<< ##
    #Cascade = cv2.CascadeClassifier(r'./eye/haarcascade_eye.xml')  #错误多  
    #Cascade = cv2.CascadeClassifier(r'./eye/haarcascade_eye_tree_eyeglasses.xml')   #好的 
    #Cascade = cv2.CascadeClassifier(r'./eye/haarcascade_lefteye_2splits.xml')    
    #Cascade = cv2.CascadeClassifier(r'./eye/haarcascade_righteye_2splits.xml')    
    ## >>>>> 人脸 <<<<<< ##
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_frontalcatface.xml') 
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_frontalcatface_extended.xml') 
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_frontalface_alt.xml') # OK
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_frontalface_alt2.xml')
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_frontalface_alt.xml')
    #Cascade = cv2.CascadeClassifier(r'./face/#haarcascade_frontalface_alt_tree.xml')
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_frontalface_default.xml')
    #Cascade = cv2.CascadeClassifier(r'./face/haarcascade_profileface.xml')   

    while cv2.waitKey(1) < 0:
        
        # get frame from the video
        hasFrame, frame = cap.read()
        
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            cv2.waitKey(3000)
            cap.release()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        objs = Cascade.detectMultiScale(gray,
                                        scaleFactor=1.15,
                                        minNeighbors=5,
                                        minSize=(5,5),
                                        flags = cv2.CASCADE_SCALE_IMAGE)
        
        for (x,y,w,h) in objs:
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
            

        # Write the frame with the detection boxes
        if (args.image):
            cv2.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv2.imshow(winName, frame)

```


## 训练经验
w h 按目标比例
storage: 看样本数量
acceptanceRatio: 在0.0004（4x10^3）左右寻找
