import os, pickle, json

def featuresAndLabels():
    modelFeature = []
    modelLabel = []
    try:
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/faceDetection"
        for fileName in os.listdir(path):
            if fileName.endswith("Features.pickle"):
                loadedModelFeatures = pickle.load(open(os.path.join(path, fileName), 'rb'))
                for eachFeature in loadedModelFeatures:
                    modelFeature.append(eachFeature)
            if fileName.endswith("Labels.pickle"):
                loadedModelLabel = pickle.load(open(os.path.join(path, fileName), 'rb'))
                for eachLabel in loadedModelLabel:
                    modelLabel.append(eachLabel)
        uniqueNames = list(set(modelLabel))
        with open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/labels.json', 'w') as f:
            json.dump(uniqueNames, f)
    except FileNotFoundError as e:
        with open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/labels.json', 'w') as f:
            json.dump(modelLabel, f)

if __name__ == "__main__":
    featuresAndLabels()
