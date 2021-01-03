
import dlib
import numpy as np
import sys
import pandas as pd
import os
import time
import shutil
import csv
import cv2
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PIL import Image, ImageDraw, ImageFont
from skimage import io


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
path_images_from_camera = "data/data_faces_from_camera/"


captureFlag = False
kk = 'p'

class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0         # 已录入的人脸计数器 
        self.ss_cnt = 0                     # 录入 personX 人脸时图片计数器 
        self.current_frame_faces_cnt = 0    # 录入人脸计数器

        self.save_flag = 1                  # 之后用来控制是否保存图像的 flag 
        self.press_n_flag = 0               
        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

    # 新建保存人脸图像文件和数据CSV文件夹 
    def pre_work_mkdir(self):
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # 删除之前存的人脸数据文件夹
    def pre_work_del_old_face_folders(self):
        # 删除之前存的人脸数据文件夹, 
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera+folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        if os.path.isfile("data/names.txt"):
            os.remove("data/names.txt")

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # 获取已录入的最后一个人脸序号 
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 
        else:
            self.existing_faces_cnt = 0

    # 获取处理之后 stream 的帧数 
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # 生成的 cv2 window 上面添加说明文字
    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face input, please keep the face in the screen, and the frame is white.", (20, 40), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), self.font, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 获取人脸 
    def process(self, stream):
        global kk
        #新建储存人脸图像文件目录 
        self.pre_work_mkdir()

        #删除 "/data/data_faces_from_camera" 中已有人脸图像文件 
        if os.path.isdir(self.path_photos_from_camera):
            self.pre_work_del_old_face_folders()

        #检查 "/data/data_faces_from_camera" 中已有人脸文件
        self.check_existing_faces_cnt()

        while stream.isOpened():
            flag, img_rd = stream.read()        
            cv2.waitKey(10)
            faces = detector(img_rd, 0)       

            if kk == ord('n'):
                self.existing_faces_cnt += 1
                current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                os.makedirs(current_face_dir)
                ui.messageList2.insertItem(0, "新建的人脸文件夹" + current_face_dir)
                self.ss_cnt = 0                 # 将人脸计数器清零
                self.press_n_flag = 1           
                kk = ord('s')

            #检测到人脸 
            if len(faces) != 0:
                for k, d in enumerate(faces):
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height/2)
                    ww = int(width/2)

                    #判断人脸矩形框是否超出 480x640 
                    if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                        cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if kk == ord('s'):
                            print("请调整位置 / Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)

                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                    if save_flag:
                        if kk == ord('s'):
                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for ii in range(height*2):
                                    for jj in range(width*2):
                                        img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                                cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                ui.messageList2.insertItem(0, "写入本地" + str(current_face_dir) + "/img_face_" + str(self.ss_cnt) + ".jpg")
                                kk = 'p'
                            else:
                                ui.messageList2.insertItem(0, "失败，请保证人脸在框中，且边框为白色")

            self.current_frame_faces_cnt = len(faces)
            self.draw_note(img_rd)
            if captureFlag:
                return
            self.update_fps()

            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)

    def run(self):
        cap = cv2.VideoCapture(0)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

class FeaturesExtraction:
    # 返回单张图像的 128D 特征
    # Input:    path_img           <class 'str'>
    # Output:   face_descriptor    <class 'dlib.vector'>
    def return_128d_features(self, path_img):
        img_rd = io.imread(path_img)
        faces = detector(img_rd, 1)

        print("%-40s %-20s" % ("检测到人脸的图像 ", path_img), '\n')

        # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            print("no face")
        return face_descriptor


    # 返回 personX 的 128D 特征均值 
    # Input:    path_faces_personX       <class 'str'>
    # Output:   features_mean_personX    <class 'numpy.ndarray'>
    def return_features_mean_personX(self, path_faces_personX):
        features_list_personX = []
        photos_list = os.listdir(path_faces_personX)
        if photos_list:
            for i in range(len(photos_list)):
                # 调用 return_128d_features() 得到 128D 特征
                print("%-40s %-20s" % ("正在读的人脸图像 / Reading image:", path_faces_personX + "/" + photos_list[i]))
                features_128d = self.return_128d_features(path_faces_personX + "/" + photos_list[i])
                # 遇到没有检测出人脸的图片跳过 
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')

        # 计算 128D 特征的均值 
        # personX 的 N 张图像 x 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=int, order='C')
        print(type(features_mean_personX))
        return features_mean_personX

    def run(self):
        # 获取已录入的最后一个人脸序号 
        person_list = os.listdir("data/data_faces_from_camera/")
        person_num_list = []
        for person in person_list:
            person_num_list.append(int(person.split('_')[-1]))
        person_cnt = max(person_num_list)

        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in range(person_cnt):
                # Get the mean/average features of face/personX, it will be a list with a length of 128D
                print(path_images_from_camera + "person_" + str(person + 1))
                features_mean_personX = self.return_features_mean_personX(path_images_from_camera + "person_" + str(person + 1))
                writer.writerow(features_mean_personX)
                print("特征均值 / The mean of features:", list(features_mean_personX))
                print('\n')
            ui.messageList2.insertItem(0, "所有录入人脸数据存入 data/features_all.csv")

class Face_Recognizer:
    def __init__(self):
        self.feature_known_list = []                # 用来存放所有录入人脸特征的数组
        self.name_known_list = []                   # 存储录入人脸名字 

        self.current_frame_face_cnt = 0             # 存储当前摄像头中捕获到的人脸数
        self.current_frame_feature_list = []        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_name_position_list = []  # 存储当前摄像头中捕获到的所有人脸的名字坐标 
        self.current_frame_name_list = []           # 存储当前摄像头中捕获到的所有人脸的名字

        self.checked_name_list = set("Person_1")

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0

    # 从 "features_all.csv" 读取录入人脸特征 
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.feature_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_"+str(i+1))
            print("Faces in Database：", len(self.feature_known_list))
            return 1
        else:
            ui.messageList1.insertItem(0, "人脸信息未找到")
            return 0

    # 计算两个128D向量间的欧式距离 
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 更新 FPS 
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        font = cv2.FONT_ITALIC

        #cv2.putText(img_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_face_cnt), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 
        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_name_list[i], self.current_frame_name_position_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_name_position_list[i], text=self.current_frame_name_list[i], font=font)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_with_name

    def show_chinese_name(self):
        # Default known name: person_1, person_2, person_3
        if self.current_frame_face_cnt >= 1:
            f = open("data/names.txt")
            line = f.readline()
            i = 0
            while line:
                self.name_known_list[i] = line.strip().encode('utf-8').decode()
                i+=1
                print(line)
                line = f.readline()
            f.close()
            faceFlag = True
            '''
            self.name_known_list[0] ='冉龙兴'.encode('utf-8').decode()
            self.name_known_list[1] ='王渝森'.encode('utf-8').decode()
            #self.name_known_list[2] ='彭红豪'.encode('utf-8').decode()
            '''
    def process(self, stream):
        #读取存放所有人脸特征的 csv 
        if self.get_face_database():
            while stream.isOpened():
                
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(10)
                if captureFlag:
                    return
                else:
                    self.draw_note(img_rd)
                    self.current_frame_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_name_position_list = []
                    self.current_frame_name_list = []

                    #检测到人脸 
                    if len(faces) != 0:
                        # 获取当前捕获到的图像的所有人脸的特征
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                        # 遍历捕获到的图像中所有的人脸
                        for k in range(len(faces)):
                            self.current_frame_name_list.append("unknown")
                            # 每个捕获人脸的名字坐标 
                            self.current_frame_name_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            # 5. 对于某张人脸，遍历所有存储的人脸特征
                            current_frame_e_distance_list = []
                            for i in range(len(self.feature_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.feature_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_feature_list[k],
                                                                                    self.feature_known_list[i])
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    current_frame_e_distance_list.append(999999999)
                            #寻找出最小的欧式距离匹配 
                            similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            if min(current_frame_e_distance_list) < 0.35:
                                self.current_frame_name_list[k] = self.name_known_list[similar_person_num]
                                if(self.name_known_list[similar_person_num] in self.checked_name_list):
                                    continue
                                else:
                                    self.checked_name_list.add(self.name_known_list[similar_person_num])
                                    ui.messageList1.insertItem(0, self.name_known_list[similar_person_num] + "   签到成功")
                                #return str(self.name_known_list[similar_person_num])
                            # 矩形框 
                            for kk, d in enumerate(faces):
                                # 绘制矩形框
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (0, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)

                        self.show_chinese_name()
                        img_with_name = self.draw_name(img_rd)

                    else:
                        img_with_name = img_rd
                cv2.imshow("camera", img_with_name)
                self.update_fps()
                
    # OpenCV 调用摄像头并进行 process
    def run(self):

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("video.mp4")
        cap.set(3, 480)     # 640x480
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()

      
Face_Recognizer = Face_Recognizer()
Face_Register_con = Face_Register()
FeaturesExtraction = FeaturesExtraction()

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(380, 369)
        self.tabWidget = QTabWidget(Form)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 381, 361))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.messageList1 = QListWidget(self.tab)
        self.messageList1.setObjectName(u"messageList1")
        self.messageList1.setGeometry(QRect(0, 0, 231, 331))
        self.startSignBtn = QPushButton(self.tab)
        self.startSignBtn.setObjectName(u"startSignBtn")
        self.startSignBtn.setGeometry(QRect(250, 10, 111, 51))
        self.stopSignBtn = QPushButton(self.tab)
        self.stopSignBtn.setObjectName(u"stopSignBtn")
        self.stopSignBtn.setGeometry(QRect(250, 90, 111, 51))
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.messageList2 = QListWidget(self.tab_2)
        self.messageList2.setObjectName(u"messageList2")
        self.messageList2.setGeometry(QRect(0, 0, 231, 331))
        self.inputNameText = QTextEdit(self.tab_2)
        self.inputNameText.setObjectName(u"inputNameText")
        self.inputNameText.setGeometry(QRect(240, 120, 131, 41))
        self.label = QLabel(self.tab_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(240, 100, 67, 17))
        self.getFaceSaveBtn = QPushButton(self.tab_2)
        self.getFaceSaveBtn.setObjectName(u"getFaceSaveBtn")
        self.getFaceSaveBtn.setGeometry(QRect(240, 50, 131, 31))
        self.startInputFaceBtn = QPushButton(self.tab_2)
        self.startInputFaceBtn.setObjectName(u"startInputFaceBtn")
        self.startInputFaceBtn.setGeometry(QRect(240, 4, 131, 31))
        self.saveNameBtn = QPushButton(self.tab_2)
        self.saveNameBtn.setObjectName(u"saveNameBtn")
        self.saveNameBtn.setGeometry(QRect(240, 170, 131, 31))
        self.closeInputFaceBtn = QPushButton(self.tab_2)
        self.closeInputFaceBtn.setObjectName(u"closeInputFaceBtn")
        self.closeInputFaceBtn.setGeometry(QRect(240, 210, 131, 31))
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(Form)

        self.tabWidget.setCurrentIndex(0)

        self.startSignBtn.clicked.connect(self.startSignFunction)
        self.stopSignBtn.clicked.connect(self.stopSignFunction)
        self.startInputFaceBtn.clicked.connect(self.startInputFaceFunction)
        self.getFaceSaveBtn.clicked.connect(self.getFaceSaveFunction)
        self.saveNameBtn.clicked.connect(self.saveNameFunction)
        self.closeInputFaceBtn.clicked.connect(self.closeInputFaceFunction)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"\u4eba\u8138\u7b7e\u5230\u7cfb\u7edf", None))
        self.startSignBtn.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb\u7b7e\u5230", None))
        self.stopSignBtn.setText(QCoreApplication.translate("Form", u"\u7ed3\u675f\u7b7e\u5230", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Form", u"\u7b7e\u5230", None))
        self.label.setText(QCoreApplication.translate("Form", u"\u8f93\u5165\u59d3\u540d\uff1a", None))
        self.getFaceSaveBtn.setText(QCoreApplication.translate("Form", u"\u83b7\u53d6\u4eba\u8138\u4fdd\u5b58", None))
        self.startInputFaceBtn.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb\u5f55\u5165\u4eba\u8138", None))
        self.saveNameBtn.setText(QCoreApplication.translate("Form", u"\u4fdd\u5b58\u59d3\u540d", None))
        self.closeInputFaceBtn.setText(QCoreApplication.translate("Form", u"\u5173\u95ed\u4eba\u8138\u5f55\u5165", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Form", u"\u4eba\u8138\u5f55\u5165", None))
    # retranslateUi

    def startSignFunction(self):
        global captureFlag
        captureFlag = False
        Face_Recognizer.run()

    def stopSignFunction(self):
        global captureFlag
        captureFlag = True
        #cv2.destroyAllWindows()

    def startInputFaceFunction(self):
        self.messageList2.insertItem(0, "开始录入人脸")
        global captureFlag
        captureFlag = False
        Face_Register_con.run()

    def getFaceSaveFunction(self):
        self.messageList2.insertItem(0, "保存人脸")
        global kk
        kk = ord('n')

    def saveNameFunction(self):
        self.messageList2.insertItem(0, "保存姓名")
        f = open("data/names.txt", "a+")
        name = self.inputNameText.toPlainText()
        f.writelines(name + "\n")
        f.close()
        self.messageList2.insertItem(0, name + "已保存")
        self.inputNameText.clear()

    def closeInputFaceFunction(self):
        self.messageList2.insertItem(0, "停止录入人脸")
        global captureFlag
        captureFlag = True
        cv2.destroyAllWindows()
        FeaturesExtraction.run()

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_()) 


    
    
