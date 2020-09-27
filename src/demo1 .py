from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import time
import xlwt
import numpy as np
import math
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *

from opts import opts
from detectors.detector_factory import detector_factory

workbook = xlwt.Workbook(encoding = "utf-8")
booksheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def get_angle1(x1,y1,x2,y2,x3,y3):
    a=math.sqrt((x2-x3)**2+(y2-y3)**2)
    b=math.sqrt((x1-x3)**2+(y1-y3)**2)
    c=math.sqrt((x2-x1)**2+(y2-y1)**2)
    if c*b==0:
      cosA=1
    else:
      cosA=(a**2-c**2-b**2)/(-2*c*b)
    if cosA < -1.0:
      cosA=-1.0
    elif cosA>1.0:
      cosA=1.0
    A=math.acos(cosA)
    deg=math.degrees(A)
    return deg

def duquwenjian(glo,points, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(17, 2)
    A8=get_angle1(points[8,0],points[8,1],points[6,0],points[6,1],points[10,0],points[10,1])
    A7=get_angle1(points[7,0],points[7,1],points[5,0],points[5,1],points[9,0],points[9,1])
    A6=get_angle1(points[6,0],points[6,1],points[8,0],points[8,1],points[12,0],points[12,1])
    A5=get_angle1(points[5,0],points[5,1],points[7,0],points[7,1],points[11,0],points[11,1])
    A12=get_angle1(points[12,0],points[12,1],points[14,0],points[14,1],points[11,0],points[11,1])
    A11=get_angle1(points[11,0],points[11,1],points[13,0],points[13,1],points[12,0],points[12,1])
    A14=get_angle1(points[14,0],points[14,1],points[16,0],points[16,1],points[12,0],points[12,1])
    A13=get_angle1(points[13,0],points[13,1],points[15,0],points[15,1],points[11,0],points[11,1])
    xcenter=(points[11,0]+points[12,0])/2
    ycenter=(points[11,1]+points[12,1])/2
    distance={}
    field=0
    for j in range(17):
      distance[j]=math.sqrt((points[j, 0]-xcenter)**2+(points[j, 1]-ycenter)**2)
      field = field +distance[j]
    print('field',field)
    #booksheet.write(glo,1,float(A8))
    #booksheet.write(glo,2,float(A7))
    #booksheet.write(glo,3,float(A6))
    #booksheet.write(glo,4,float(A5))
    #booksheet.write(glo,5,float(A12))
    #booksheet.write(glo,6,float(A11))
    #booksheet.write(glo,7,float(A14))
    #booksheet.write(glo,8,float(A13))
    #booksheet.write(glo,9,float(field))


    booksheet.write(glo,1,float(points[5,0]))
    booksheet.write(glo,2,float(points[5,1]))
    booksheet.write(glo,3,float(points[6,0]))
    booksheet.write(glo,4,float(points[6,1]))
    booksheet.write(glo,5,float(points[7,0]))
    booksheet.write(glo,6,float(points[7,1]))
    booksheet.write(glo,7,float(points[8,0]))
    booksheet.write(glo,8,float(points[8,1]))
    booksheet.write(glo,9,float(points[9,0]))
    booksheet.write(glo,10,float(points[9,1]))
    booksheet.write(glo,11,float(points[10,0]))
    booksheet.write(glo,12,float(points[10,1]))
    booksheet.write(glo,13,float(points[11,0]))
    booksheet.write(glo,14,float(points[11,1]))
    booksheet.write(glo,15,float(points[12,0]))
    booksheet.write(glo,16,float(points[12,1]))
    booksheet.write(glo,17,float(points[13,0]))
    booksheet.write(glo,18,float(points[13,1]))
    booksheet.write(glo,19,float(points[14,0]))
    booksheet.write(glo,20,float(points[14,1]))
    booksheet.write(glo,21,float(points[15,0]))
    booksheet.write(glo,22,float(points[15,1]))
    booksheet.write(glo,23,float(points[16,0]))
    booksheet.write(glo,24,float(points[16,1]))
   


class VideoBox(QWidget):

    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    video_url = ""

    def __init__(self, video_url="", video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        QWidget.__init__(self)
        self.video_url = video_url
        self.video_type = video_type  # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause

        # 组件展示
        self.pictureLabel = QLabel()
        init_image = QPixmap("11.jpeg").scaled(self.width(), self.height())
        self.pictureLabel.setPixmap(init_image)

        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)

        control_box = QHBoxLayout()
        control_box.setContentsMargins(0, 0, 0, 0)
        control_box.addWidget(self.playButton)

        layout = QVBoxLayout()
        layout.addWidget(self.pictureLabel)
        layout.addLayout(control_box)

        self.setLayout(layout)

        # timer 设置
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)

        # video 初始设置
        self.playCapture = VideoCapture()
        if self.video_url != "":
            self.playCapture.open(self.video_url)
            fps = self.playCapture.get(CAP_PROP_FPS)
            self.timer.set_fps(fps)
            self.playCapture.release()
            if self.auto_play:
                self.switch_video()
            # self.videoWriter = VideoWriter('*.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, size)

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoBox.STATUS_INIT
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def show_video_images(self):
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            frame=detector.run(frame)
            if success:
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cvtColor(frame, COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cvtColor(frame, COLOR_GRAY2BGR)
                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                # print(type(rgb))
                temp_pixmap = QPixmap.fromImage(temp_image)
                print(type(temp_pixmap))

                self.pictureLabel.setPixmap(temp_pixmap)
            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()
                if not success and self.video_type is VideoBox.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # 判断本地文件播放完毕
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def switch_video(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.status is VideoBox.STATUS_INIT:
            self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.status is VideoBox.STATUS_PLAYING:
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        elif self.status is VideoBox.STATUS_PAUSE:
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        self.status = (VideoBox.STATUS_PLAYING,
                       VideoBox.STATUS_PAUSE,
                       VideoBox.STATUS_PLAYING)[self.status]


class Communicate(QObject):

    signal = pyqtSignal(str)


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stoppedQWidget

    def set_fps(self, fps):
        self.frequent = fps


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  app = QApplication(sys.argv)
  box = VideoBox("haojiang.mp4")
  box.show()
  sys.exit(app.exec_())
'''
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    #cam = cv2.VideoCapture("../images/webwxgetvideo")
    detector.pause = False
    t=0
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('differ.avi',codec,25.0,size)
    while cam.isOpened():
        ret, img = cam.read()

        cv2.imshow('input', img)
        s=time.time()
        ret = detector.run(img)
        results = ret['results']
        for bbox in results[1]:
            if bbox[4] > opt.vis_thresh:
                duquwenjian(t,bbox[5:39],img_id='multi_pose')
        e=time.time()
        fps=1/(e-s)
        print("FPS",fps)
        #print("FPS",1/ret['tot'])
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if t in [0,1,39,87,88,177,178,180,181,182,184,195,196,200,201,202,203,204,205,206,207,208,212,217,222,223,229,230,239,240,241,257,258,259,285,286,343,346,347,348,349,350,351,352,353,358,359,360,366,371,379,386,387,388]:
            output.write(img)
        t=t+1
        #print(t)
        if cv2.waitKey(10) & 0XFF == ord('q'):
            print("ed")
            workbook.save("dance22.xls")
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()
    workbook.save("test.xls")
    
    #stream.release()
    
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      print(ret)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
'''
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
