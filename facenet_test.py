# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:45:49 2020

@author: sshss
"""

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
warnings.filterwarnings('ignore')


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
    def __init__(self, size, margin, weights, device, select_largest):

        print("Loading network.....")
        current = time.time()
        self.device = device
        self.mtcnn = MTCNN(image_size=size, margin=margin, keep_all=False, device=device, select_largest=True)
        self.model = InceptionResnetV1(pretrained='vggface2',device=device)
        state_dict = torch.load(weights)
        self.model.load_state_dict(state_dict)
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
        name = input("please enter your name:\n")
        img_cropped,prob = self.mtcnn(img, return_prob=True)
        if img_cropped is not None:
            boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
            # draw bbox in image
            dst = draw_boxes(img, boxes, probs)
            img_cropped = img_cropped.to(self.device)
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = self.model(img_cropped.unsqueeze(0)).to('cpu')
            return name, img_embedding, dst
        else:
            print("No face detected, Try again!!!")
            return None, None, None
        
    def recognize(self, img, threshold):
        img_cropped,prob = self.mtcnn(img, return_prob=True)
        if img_cropped is not None:
            boxes, probs, points = self.mtcnn.detect(img, landmarks=True)
            img_cropped = img_cropped.to(self.device)
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = self.model(img_cropped.unsqueeze(0)).to('cpu')
            smallest_dist = 999
            embs = glob.glob(osp.realpath(args.images)+'\\*.embs')
            for emb in embs:
                dist = calc_dist(img_embedding.detach().numpy(), emb2numpy(emb))
                if smallest_dist >  dist:
                    smallest_dist = dist
                    recog_name = emb.split('\\')[-1].split('.')[0]
            if smallest_dist > threshold:
                print('recognize unreliable!')
            else:
                print("#"*50)
                print('Staff identified! Welcome: {}'.format(recog_name))
                print("login time is: {}".format(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
                print("#"*50)
                
            dst = draw_boxes(img, boxes, recog_name)
            cv2.putText(dst, str(smallest_dist), (10,710), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
            return recog_name, dst
        else:
            print("No face detected, Try again!!!")
            return None, None



if __name__ == "__main__":
    args=arg_parse()
    flag = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mode = args.mode
    objects = args.objects
    margin = args.margin
    size = args.size
    weights = args.weights
    select_largest = args.select_largest
    dist_thresh = args.dist_thresh
    det = osp.realpath(args.det)
    db = osp.realpath(args.images)
    if not osp.exists(det):
        os.mkdir(det)
    if not osp.exists(db):
        os.mkdir(db)
     
    face = faceapp(size, margin, weights, device, select_largest)
    if mode == 'recognize':
        while flag:
            img = capture()
            result_name, dst = face.recognize(img, dist_thresh)
            if result_name is None:
                continue
            else:
                flag=0          
        cv2.imwrite(osp.join(det,'%s_det.jpg'%result_name),dst)
        
    elif mode == 'collect':
        while flag:         
            img = capture()
            user_name ,img_embedding, dst = face.collect(img)
            if user_name is None:
                continue
            else:
                flag=0 
        cv2.imwrite(osp.join(db,'%s_cap.jpg'%user_name),dst)
        joblib.dump(img_embedding, osp.join(db, "%s.embs"%user_name))
    elif mode == 'realtime':
        face.realtime()
        

        

