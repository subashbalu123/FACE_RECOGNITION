from datetime import datetime
import cv2
import numpy as np
import json
import pytesseract
import re
import requests
from datetime import datetime

json_values = {}


class PlateFinder:
    def __init__(self):
        self.min_area = 0  # minimum area of the plate
        self.max_area = 0  # maximum area of the plate

        # self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(self.ksize1, self.ksize2))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)  # old window was (5,5)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)  # convert to gray
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # sobelX to get the vertical edges
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        contours, hierarchy = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:

            areas = [cv2.contourArea(c) for c in contours]
            # print((areas))
            max_index = np.argmax(areas)  # index of the largest contour in the area array

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = plate
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)

            if plateFound:
                x1, y1, w1, h1 = coordinates
                coordinates = x1 + x, y1 + y
                after_check_plate_img = after_clean_plate_img
                return after_check_plate_img, coordinates
        return None, None

    def find_possible_plates(self, input_img):
        """
        Finding all possible contours that can be plates
        """
        plates = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)
        for cnts in possible_plate_contours:
            plate, coordinates = self.check_plate(input_img, cnts)

            if plate is not None:
                plates.append(plate)
                self.corresponding_area.append(coordinates)

        if len(plates) > 0:
            return plates
        else:
            return None

    # PLATE FEATURES
    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 3
        ratioMax = 6

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def preRatioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 2.5
        ratioMax = 7

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect

        if width > height:
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 15:
            return False
        if height == 0 or width == 0:
            return False

        area = width * height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True

def empty(a):
    pass


def detect(ksize1, ksize2, minimum_area, maximum_area, frame, postURL, place, cameraId, fps):
    findPlate = PlateFinder()
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
    img = frame
    findPlate.min_area = minimum_area
    findPlate.max_area = maximum_area
    findPlate.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(ksize1, ksize2))
    count = 0
    possible_plates = findPlate.find_possible_plates(img)
    if possible_plates is not None:
        plate = possible_plates[0]
        #print("-------------------")
        img_src = cv2.imread(plate)
        gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(img_src, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = img_src[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            print("PLATE NUMBER=", text)
        text = pytesseract.image_to_string(plate, config='--psm 11')
        reg_exp1 = '\w{2}\d{2}\w{2}\d{4}'
        reg_exp2 = '\w{2} \d{2} \w{2} \d{4}'
        reg_exp3 = '\w{2} \d{2}\w{2} \d{4}'
        reg_exp4 = '\w{2}\d{2} \w{2}\d{4}'
        reg_exp5 = '\w{2}\d{2}\w{2} \d{4}'
        reg_exp6 = '\w{2} \d{2}\w{2}\d{4}'
        reg_exp7 = '\w{2} \d{2}\w{2} \d{4}'
        reg_exp8 = '\w{2}.\d{2}.\w{2}.\d{4}'
        reg_exp9 = '\w{2} .\d{2} .\w{2} .\d{4}'
        reg_exp10 = '\w{2} .\d{2}.\w{2} .\d{4}'
        reg_exp11 = '\w{2}. \d{2}.\w{2}. \d{4}'
        reg_exp13 = '\w{2}\d{2}\w{1}.\d{4}'
        reg_exp14 = '\w{2}\d{2} \w{2} \d{4}'

        if (re.search(reg_exp1, text) or re.search(reg_exp2, text) or re.search(reg_exp3, text) or
                re.search(reg_exp4, text) or re.search(reg_exp5, text) or re.search(reg_exp6, text) or
                re.search(reg_exp7, text) or re.search(reg_exp8, text) or re.search(reg_exp9, text) or
                re.search(reg_exp10, text) or re.search(reg_exp11, text) or re.search(reg_exp13, text) or
                re.search(reg_exp14, text)):
            print("Detected Number is:", text)
            json_values["channelId"] = cameraId
            json_values["channelName"] = place
            json_values["entryExit"] = None
            json_values["entryViolationVos"] = None
            json_values["npr"] = {"values": ["string"]}  # here values need
            json_values["snapshot"] = None
            json_values["socialViolation"] = None
            json_values["time"] = dt_string
        try:
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'CLIENT_KEY': 'ilens client key'}
            response = requests.post(url=postURL, data=json.dumps(json_values), headers=headers, verify=False)
            print("Status:", response.status_code)
        except BaseException as e:
            print("Status:", e)
        json_values.clear()
    print(f'fps at npr, {int(fps)}')
