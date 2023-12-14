#camera.py
# import the necessary packages
import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

model = YOLO("yolov8n.pt")
counter = object_counter.ObjectCounter()  # Init Object Counter
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)


# defining face detector
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
        #capturing video
        #    self.video = cv2.VideoCapture(0)
        # self.video = cv2.VideoCapture(0,cv2.CAP_MSMF)
        self.video = cv2.VideoCapture("./data/worker-zone-detection.mp4")

    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
        #extracting frames
        ret, raw_frame = self.video.read()

        tracks = model.track(raw_frame, persist=True, show=False)
        frame = counter.start_counting(raw_frame, tracks)

        if frame is not None:
            frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
            interpolation=cv2.INTER_AREA)                    
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
            # for (x,y,w,h) in face_rects:
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                # break

            # encode OpenCV raw frame to jpg and displaying it
            ret, jpeg = cv2.imencode('.jpg', frame)
        else:
            ret, jpeg = cv2.imencode('.jpg', raw_frame)
        return jpeg.tobytes()