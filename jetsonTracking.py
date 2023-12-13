import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 

timeStamp=time.time()
fpsFilt=0
net = jetson.inference.detectNet(argv=["--model=ssd-mobilenet.onnx",
                      "--labels=labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"]
                , threshold=0.8)
net.SetTrackerType("IOU")
net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3,dropFrames=15,overlapThreshold=0.8)


dispW=640
dispH=480

font=cv2.FONT_HERSHEY_SIMPLEX

roi_bound=dispH/2

in_count=0
out_count=0

prevPosition={}

cam=cv2.VideoCapture('/dev/video0')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)


while True:
   
    _,img = cam.read()
    height=img.shape[0]
    width=img.shape[1]

    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    frame=jetson.utils.cudaFromNumpy(frame)

    detections=net.Detect(frame, width, height)
    for detect in detections:
        #print(detect)
        ID=detect.ClassID
        top=detect.Top
        left=detect.Left
        bottom=detect.Bottom
        right=detect.Right
        item=net.GetClassDesc(ID)
        track_id=detect.TrackID
        currentCentroid=detect.Center[1]
        # # print(item,top,left,bottom,right)

        if track_id !=-1:
            if track_id in prevPosition:
                previousCentroid=prevPosition[track_id]
                if previousCentroid < roi_bound and currentCentroid >= roi_bound:
                    in_count+=1
                    # add firebase function here please use dict
                elif previousCentroid > roi_bound and currentCentroid <= roi_bound:
                    out_count+=1
                    # add firebase function here please use dict
            prevPosition[track_id]=currentCentroid


        cv2.rectangle(img,(int(left),int(top)),(int(right),int(bottom)),(255,0,0),thickness=2)
        cv2.putText(img,item,(int(left),int(top)),font,1,(255,0,0),2)
    #display.RenderOnce(img,width,height)
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps
    #print(str(round(fps,1))+' fps')
    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
    cv2.putText(img,'In: '+str(in_count),(0,60),font,1,(0,0,255),2)
    cv2.putText(img,'Out: '+str(out_count),(0,90),font,1,(0,0,255),2)
    cv2.imshow('detCam',img)
    cv2.moveWindow('detCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()