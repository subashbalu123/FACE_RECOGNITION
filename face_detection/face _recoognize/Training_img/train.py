import cv2, os, pickle, numpy as np
from face_recognition import face_encodings, face_locations
from cvzone.SelfiSegmentationModule import SelfiSegmentation
segmentor = SelfiSegmentation()


def adjustGamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


class Train:
    def __init__(self, basePath, imagePath, userName):
        self.basePath = basePath
        self.imagePath = imagePath
        self.userName = userName

    def dataTrain(self):
        try:
            classNames = []
            encodeList = []
            baseDirectory = os.path.dirname(os.path.abspath(__file__))
            imageDirectory = os.path.join(baseDirectory, self.imagePath)
            for root, directory, files in os.walk(imageDirectory):
                for file in files:
                    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                        path = os.path.join(root, file)
                        label = os.path.basename(root).replace(" ", "_").lower()
                        img = cv2.imread(path)
                        # img = segmentor.removeBG(img, (206, 207, 204), 0.5)
                        # img = cv2.resize(img, (0, 0), fx=0.60, fy=0.60)
                        img = adjustGamma(img, gamma=1.7)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        faces = face_locations(img)
                        try:
                            encode = face_encodings(img, faces, model="large")[0]
                            encodeList.append(encode)
                            classNames.append(label)
                            print(encode)
                        except IndexError as index:
                            continue
            filename = self.basePath + "/" + self.userName + "Features.pickle"
            if os.path.exists(os.path.join(self.basePath, filename)):
                unPicklingValues = pickle.load(open(filename, 'rb'))
                encodeList = unPicklingValues + encodeList
            pickle.dump(encodeList, open(filename, 'wb'))

            filename_label = self.basePath + "/" + self.userName + "Labels.pickle"
            if os.path.exists(os.path.join(self.basePath, filename_label)):
                unPicklingValues = pickle.load(open(filename_label, 'rb'))
                classNames = unPicklingValues + classNames
            pickle.dump(classNames, open(filename_label, 'wb'))
            print("[INFO]: Face Data Training Completed.")
        except BaseException as e:
            raise BaseException("Data Training Error {}", e)
