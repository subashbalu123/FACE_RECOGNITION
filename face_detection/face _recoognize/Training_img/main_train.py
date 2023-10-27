import cv2
import os
from train import Train
import argparse


class MainTrain:
    def __init__(self, employeeId, basePath, imagePath, userName):
        self.employeeId = employeeId
        self.basePath = basePath
        self.imagePath = imagePath
        self.userName = userName

    def dataTraining(self):
        # face_cascade = cv2.CascadeClassifier(
        #     self.basePath + "/face_training/training_imgs/haar_cascad_frontalface_detection.xml")
        try:
            print("[INFO]: Image Processing Started...")
            basePath = self.basePath + "/face_training/training_imgs"
            directory = os.path.join(basePath, self.employeeId)
            for filename in os.listdir(self.imagePath):
                if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    path = os.path.join(self.imagePath, filename)
                    img = cv2.imread(path)
                    shapeValues = img.shape
                    """ if shapeValues[0] <= 800 and shapeValues[1] <= 370:
                        print("Entered inside")
                        dimensions = (800, 470)
                        img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA) """
                    # faces = face_cascade.detectMultiScale(img, 1.3, 5)
                    # for (x, y, w, h) in faces:
                    #     roi_color = img[y:y + h, x:x + w]
                    fileName = filename[:len(filename) - 4]
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                    cv2.imwrite(directory + "/" + self.employeeId + fileName + ".jpg", img)
            print("[INFO]: Image Processing Completed.")
            print("[INFO]: Data training started...")
            trainObj = Train(self.basePath, directory, self.userName)
            trainObj.dataTrain()
        except BaseException as e:
            raise Exception(e)


if __name__ == "__main__":
    argumentInput = argparse.ArgumentParser()
    argumentInput.add_argument("-i", "--employeeId", required=True, help="employee_id")
    argumentInput.add_argument("-b", "--basePath", required=True, help="base_path/FaceDetection is basepath")
    argumentInput.add_argument("-p", "--imgPath", required=True, help="image_path_from_api")
    argumentInput.add_argument("-u", "--sessionUserName", required=True, help="session user name for pickling")
    inputData = argumentInput.parse_args()
    MainTrainObj = MainTrain(inputData.employeeId, inputData.basePath, inputData.imgPath, inputData.sessionUserName)
    MainTrainObj.dataTraining()
