import os, cv2, json, pickle, requests, urllib3, numpy as np
from datetime import datetime
from face_recognition import face_encodings, face_locations

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + "/userList.json", "r")
    data = json.loads(f.read())
except IOError as e:
    raise IOError("Users list not found", e)


def adjustGamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def checkContiguousOccurrence(matches, distances, labels):
    maxCount, currentCount = 0, 0
    max_matches, current_matches, max_index, current_index, distance, labelsLst = [], [], [], [], [], []
    for i in range(len(matches)):
        if matches[i] == 'True':
            currentCount += 1
            current_matches.append(matches[i])
            current_index.append(i)
            if i == len(matches) - 1:
                maxCount = max(currentCount, maxCount)
                if len(current_matches) == maxCount:
                    max_matches = current_matches
                    max_index = current_index
        else:
            maxCount = max(currentCount, maxCount)
            if len(current_matches) == maxCount:
                max_matches = current_matches
                max_index = current_index
            currentCount = 0
            current_matches = []
            current_index = []

    for ind in max_index:
        distance.append(distances[ind])
        labelsLst.append(labels[ind])
    return maxCount, max_matches, distance, labelsLst


def featuresAndLabels():
    modelFeature = []
    modelLabel = []
    path = os.path.dirname(os.path.abspath(__file__))
    for fileName in os.listdir(path):
        if fileName.endswith("Features.pickle"):
            loadedModelFeatures = pickle.load(open(os.path.join(path, fileName), 'rb'))
            for eachFeature in loadedModelFeatures:
                modelFeature.append(eachFeature)
        if fileName.endswith("Labels.pickle"):
            loadedModelLabel = pickle.load(open(os.path.join(path, fileName), 'rb'))
            for eachLabel in loadedModelLabel:
                modelLabel.append(eachLabel)
    print(f'Labels: {modelLabel}')
    return modelFeature, modelLabel


# trained Features and Labels
modelFeatures, modelLabels = featuresAndLabels()
unknownEncodings = []
CONFIDENCE = .5
resizeRatio = 50


def postUnknown(dataLocation, dt_string, frame, apiToken, postURL, json_values):
    face_file_name = "".join([dataLocation, "/unknown/", dt_string, ".jpg"])
    cv2.imwrite(face_file_name, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (490, 334),
                                           interpolation=cv2.INTER_AREA))
    try:
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain',
                   'CLIENT_KEY': str(apiToken)}
        response = requests.post(url=postURL + "/unknown/save", data=json.dumps(json_values),
                                 headers=headers, verify=False)
        print("[INFO]: Unknown Person : {}", response.status_code)
    except ConnectionError as e:
        raise ConnectionError("Server Connection Exception {}", e)


class FRMethod:
    def __init__(self, frame, basePath, cameraId, place, entryOrExit, postURL, dataLocation, apiToken, startTime):
        self.frame = frame
        self.basePath = basePath
        self.cameraId = cameraId
        self.place = place
        self.entryOrExit = entryOrExit
        self.postURL = postURL
        self.dataLocation = dataLocation
        self.apiToken = apiToken
        self.startTime = startTime

    def liveMethod(self):
        global unknownEncodings
        now, json_values, emp_name = datetime.now(), {}, "----"
        dt_string = now.strftime("%d%m%Y%H%M%S")

        # brightness adjustment.
        img = adjustGamma(self.frame, gamma=1.7)

        # resize part.
        dim = (int(img.shape[1] * resizeRatio / 100), int(img.shape[0] * resizeRatio / 100))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        faces = face_locations(img)
        diff = datetime.now() - self.startTime
        if len(faces) != 0:
            encodesCurFrame = face_encodings(img, faces, model="large")
            for encodeFace, faceLoc in zip(encodesCurFrame, faces):  # comparison
                matches = (np.linalg.norm(modelFeatures - encodeFace, axis=1) <= CONFIDENCE)
                faceDistance = np.linalg.norm(modelFeatures - encodeFace, axis=1)
                matchList = [f'{str(i)}' for i in matches]
                contCount, matches, faceDistance, labels = checkContiguousOccurrence(matchList, faceDistance, modelLabels)
                print(f'contCount, {contCount}')
                (top, right, bottom, left) = faceLoc
                json_values["channelId"] = self.cameraId
                json_values["channelName"] = self.place
                json_values["time"] = dt_string
                json_values["type"] = self.entryOrExit
                if contCount >= 2:  # based on Training image quantity
                    matchIndex = np.argmin(faceDistance)
                    if matches[matchIndex]:
                        emp_id = labels[matchIndex]
                        if data['usersList'].get(emp_id): emp_name = data['usersList'].get(emp_id)
                        face_file_name = "".join([self.dataLocation, "/", dt_string + "_" + emp_id, ".jpg"])
                        json_values["snapshot"] = dt_string + "_" + emp_id
                        json_values["entryExit"] = [{"id": emp_id, "name": emp_name}]
                        json_values["entryViolationVos"] = None
                        json_values["npr"] = None
                        json_values["socialViolation"] = None
                        cv2.rectangle(img, (left, top), (right, bottom), (252, 3, 3), 2)
                        cv2.putText(img, emp_name, (left - 10, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (252, 3, 3), 1, cv2.LINE_AA)
                        cv2.imwrite(face_file_name, cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (490, 334),
                                                               interpolation=cv2.INTER_AREA))
                        try:
                            headers = {'Content-type': 'application/json', 'Accept': 'text/plain',
                                       'CLIENT_KEY': str(self.apiToken)}
                            response = requests.post(url=self.postURL + "/dataset", data=json.dumps(json_values),
                                                     headers=headers, verify=False)
                            diff1 = datetime.now() - self.startTime
                            print()
                            print("[INFO]: Captured Information Post Response : {}", response.status_code)
                        except ConnectionError as e:
                            raise ConnectionError("Server Connection Exception {}", e)
            #
            #                 else:
            #                     json_values["snapshot"] = dt_string
            #                     if unknownEncodings is None or len(unknownEncodings) == 0:
            #                         try:
            #                             postUnknown(self.dataLocation, dt_string, img, self.apiToken,
            #                             self.postURL, json_values)
            #                             unknownEncodings = [encodeFace]
            #                         except ConnectionError as e:
            #                             raise ConnectionError("Server Connection Exception {}", e)
            #                     else:
            #                         checkConfidence = np.linalg.norm(unknownEncodings - encodeFace) < CONFIDENCE
            #                         if not checkConfidence:
            #                             try:
            #                                 postUnknown(self.dataLocation, dt_string, img, self.apiToken,
            #                                 self.postURL, json_values)
            #                                 unknownEncodings = [encodeFace]
            #                             except ConnectionError as e:
            #                                 raise ConnectionError("Server Connection Exception {}", e)
                json_values.clear()
        # else:
        #     # cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        #     # cv2.putText(img, 'unknown', (left - 10, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
        #     #            2, cv2.LINE_AA)
        #     face_file_name = "".join([self.dataLocation, "/", dt_string, ".jpg"])
        #     cv2.imwrite(face_file_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # print(f'{json_values}')
        # print(f'fps at fr, {int(self.fps)}')
        # json_values.clear()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.putText(img, f'fps : {int(self.fps)}', (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.imshow("Output Image", img)
        # cv2.waitKey(1)
