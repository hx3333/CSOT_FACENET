#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2020年10月12日
@author: Ted
@email: 
@description: simple ui for facerecognition
"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QPlainTextEdit,QInputDialog,QMessageBox
from PyQt5 import QtCore
from facenet import MTCNN, InceptionResnetV1
from sklearn.externals import joblib
from scipy.spatial import distance
import torch
import cv2, os, time, glob, datetime, random
import numpy as np
import torch.nn.functional as F
import argparse
import warnings
import os.path as osp
import pickle as pkl
from time import clock
from enum import Enum
warnings.filterwarnings('ignore')

__Author__ = "Ted"
__Copyright__ = ""
__Version__ = ""


class ErrEnum(Enum):
    No_problem = 0
    No_face_detected = 1
    Unreliable_recognize = 2 

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='Facenet_Pytorch Module')
   
    parser.add_argument("--images", help = "Image / Directory to store face images",
                        default = "samples", type = str)
    parser.add_argument("--det", help = "Image / Directory to view face-detections result ",
                        default = "det", type = str)
    parser.add_argument("--mode", help = "collect, recognize, realtime ",
                        default = "realtime", type = str)
    parser.add_argument("--dist_thresh", help = "threshold of euclidean distance",
                        default = 0.75, type = float)
    parser.add_argument("--weights", help = "weightsfile",
                        default = "20180402-114759-vggface2.pt", type = str)
    parser.add_argument("--objects", help = "specified if to detect multi-faces",
                        default = True, type = bool)
    parser.add_argument("--margin", help = "add more(or less) of a margin around the faces",
                        default = 0, type = int)
    parser.add_argument("--size", help = "specified the size of faces",
                        default = 160, type = int)
    parser.add_argument("--select_largest", help = "collect embedding feature set True",
                        default = True, type = bool)
    parser.add_argument("--timeout", help = "timeout for rec or record",
                        default = 30, type = int)
    
    return parser.parse_args()

def calc_dist(src, emb):
    dist = distance.euclidean(src, emb)
    return dist

def emb2numpy(src):
    src = joblib.load(src)
    try:    
        src_l2_norm=F.normalize(src).detach().numpy()
    except TypeError:
        src_l2_norm=F.normalize(src).cpu().detach().numpy()
    return src_l2_norm

def draw_boxes(img, boxes, prob, pallete='pallete'):
    colors = pkl.load(open(pallete,'rb'))
    color = random.choice(colors)
    img = img.copy()
    for i in range(len(boxes)):
        c1 = tuple(np.rint(boxes[i][0:2]).astype('int32'))
        c2 = tuple(np.rint(boxes[i][2:4]).astype('int32'))
        cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(str(prob[i]), cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.putText(img, str(prob[i]), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def capture():
    label = "press q to quit, image will save automatically"
    cap = cv2.VideoCapture(0)
    # set webcam resolution
    cap.set(3, 1280)
    cap.set(4, 720)
    prev_frame_time = 0 
    new_frame_time = 0  
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            break
        font = cv2.FONT_HERSHEY_PLAIN
        new_frame_time = time.time()
        try:
            fps = 1/(new_frame_time-prev_frame_time)
        except ZeroDivisionError:
            continue
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, "FPS:%s"%fps, (10,25), font, 1, [0,0,255], 1)
        cv2.putText(frame, label, (10,10), font, 1, [0,0,255], 1)
        cv2.imshow("capture",frame)
        if cv2.waitKey(1)&0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


class faceapp():
    def __init__(self, size, margin, weights, device, select_largest,images):

        print("Loading network.....")
        current = time.time()
        self.device = device
        self.mtcnn = MTCNN(image_size=size, margin=margin, keep_all=False, device=device, select_largest=True)
        self.model = InceptionResnetV1(pretrained='vggface2',device=device)
        state_dict = torch.load(weights,map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.images=images
        elapse = time.time()-current
        print("Network successfully loaded, process time: %.2fs"%elapse)
        # eval mode
        self.model.eval()
        
    def realtime(self):
        cap = cv2.VideoCapture(0)  
        cap.set(3, 1280)
        cap.set(4, 720)
        prev_frame_time = 0 
        new_frame_time = 0  
        assert cap.isOpened(), 'Cannot capture source' 
        frames = 0
        start = time.time()    
        while cap.isOpened():    
            ret, frame = cap.read()
            if ret:
                boxes, probs, points = self.mtcnn.detect(frame, landmarks=True)
                if boxes is None:
                    continue
                dst = draw_boxes(frame, boxes, probs)
                font = cv2.FONT_HERSHEY_PLAIN
                new_frame_time = time.time()
                try:
                    fps = 1/(new_frame_time-prev_frame_time)
                except ZeroDivisionError:
                    continue
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(dst, "FPS:%s"%fps, (10,25), font, 1, [0,0,255], 1)
                cv2.imshow('temp',dst)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        

    def collect(self, img):
        img_cropped,prob = self.mtcnn(img, return_prob=True)
        if img_cropped is not None:
            boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
            # draw bbox in image
            dst = draw_boxes(img, boxes, probs)
            img_cropped = img_cropped.to(self.device)
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = self.model(img_cropped.unsqueeze(0)).to('cpu')
            return  ErrEnum.No_problem,img_embedding, dst
        else:
            print("No face detected, Try again!!!")
            return  ErrEnum.No_face_detected,None, None
        
    def recognize(self, img, threshold):
        img_cropped,prob = self.mtcnn(img, return_prob=True)
        if img_cropped is not None:
            boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
            img_cropped = img_cropped.to(self.device)
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = self.model(img_cropped.unsqueeze(0)).to('cpu')
            smallest_dist = 999
            embs = glob.glob(osp.realpath(self.images)+'\\*.embs')
            for emb in embs:
                dist = calc_dist(img_embedding.detach().numpy(), emb2numpy(emb))
                if smallest_dist >  dist:
                    smallest_dist = dist
                    recog_name = emb.split('\\')[-1].split('.')[0]
            if smallest_dist > threshold:
                # print('recognize unreliable!')
                return ErrEnum.Unreliable_recognize ,"", None
            else:
                print("#"*50)
                print('Staff identified! Welcome: {}'.format(recog_name))
                print("login time is: {}".format(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
                print("#"*50)
                
            dst = draw_boxes(img, boxes, recog_name)
            cv2.putText(dst, str(smallest_dist), (10,710), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
            return ErrEnum.No_problem, recog_name, dst
        else:
            print("No face detected, Try again!!!")
            return ErrEnum.No_face_detected, None, None


class Window(QWidget):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        # self.setWindowOpacity(0.9) 
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground) 
        self.setWindowTitle("face")
        layout = QHBoxLayout(self)
        layout.addWidget(QPushButton(
            'Record', self, objectName='Recordbtn', pressed=self.onRecord))
        layout.addWidget(QPushButton(
            'Recognize', self, objectName='Recognizebtn', pressed=self.onRecog))

        self.resultView = QPlainTextEdit(self)
        self.resultView.setReadOnly(True)
        layout.addWidget(self.resultView)
        # start
        args=arg_parse() # argparse
        print(args)
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        mode = args.mode
        objects = args.objects
        margin = args.margin
        size = args.size
        weights = args.weights
        images = args.images
        self.timeout=args.timeout
        select_largest = args.select_largest
        self.dist_thresh = args.dist_thresh
        self.det = osp.realpath(args.det)
        self.db = osp.realpath(args.images)
        if not osp.exists(self.det):
            os.mkdir(self.det)
        if not osp.exists(self.db):
            os.mkdir(self.db)
         
        self.face = faceapp(size, margin, weights, device, select_largest,images)
        
        # self.setStyleSheet('''
        # QPushButton{border:none;color:white;}
        # QPushButton#left_label{
            # border:none;
            # border-bottom:1px solid white;
            # font-size:18px;
            # font-weight:700;
            # font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        # }
        # QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
        # ''')

    def handleErr(self,err):
        if err==ErrEnum.No_face_detected:
            QMessageBox.warning(self,"Warning","No face detected, Try again!!!")
            self.resultView.appendPlainText("No face detected...\n")
        elif err==ErrEnum.Unreliable_recognize:
            QMessageBox.warning(self,"Warning","recognize unreliable!")
            self.resultView.appendPlainText("recognize unreliable!...\n")
            
    def onRecog(self):
        
        flag = 1
        start=clock()
        self.resultView.appendPlainText("start recognizing...\n")
        while flag:
            img = capture()
            err, result_name, dst = self.face.recognize(img, self.dist_thresh)
            if err != ErrEnum.No_problem:
                self.handleErr(err)
                
                return
            if result_name is None:
                continue
            else:
                flag=0 
            QMessageBox.information(self,"Info","Staff identified! Welcome: {} \nlogin time is: {}".format(result_name,datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        cv2.imwrite(osp.join(self.det,'%s_det.jpg'%result_name),dst)

    def onRecord(self):
        self.resultView.appendPlainText("start recording...\n")
        user_name,bret = QInputDialog.getText(self,"Name","please enter your name:\n")
        if not bret:
            QMessageBox.warning(self,"Warning","No username input...")
            self.resultView.appendPlainText("username empty operation canceled \n")
            return
        self.resultView.appendPlainText("username: {} \n".format(user_name))
        flag = 1
        start=clock()
        while flag:         
            img = capture()
            err, img_embedding, dst = self.face.collect(img)
            if img_embedding is None:
                if err != ErrEnum.No_problem:
                    self.handleErr(err)
                    return
                end=clock()
                if end-start>self.timeout :
                    self.resultView.appendPlainText("operation timeout...\n")
                    return
            else:
                
                flag=0 
        cv2.imwrite(osp.join(self.db,'%s_cap.jpg'%user_name),dst)
        joblib.dump(img_embedding, osp.join(self.db, "%s.embs"%user_name))



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())
