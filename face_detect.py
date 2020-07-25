# from mtcnn.mtcnn import MTCNN
# import matplotlib.pyplot as plt
#
#
#
# image = "image1.png"
#
#
# def face_detect(input_img):
#     model = MTCNN()
#     img = plt.imread(input_img)
#     faces = model.detect_faces(img)
#
#     if len(faces) != 0:
#         print(face for face in faces)
#     else:
#         print("failed to find face")
#
#
# face_detect(image)


# face detection with mtcnn on a photograph
from matplotlib import pyplot
import os
import csv
from matplotlib.patches import Rectangle
import mysql.connector
import sqlite3
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine



def detect_faces(filename):
    filename = "/Users/nisha/Downloads/Test_set/" + filename
    #convert png to jpg
    if filename[-4:].lower() == ".png":
        img = Image.open(filename)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        filename = filename[:-4] + ".jpg"
        img.save(filename)

    img = pyplot.imread(filename)
    model = MTCNN()
    # returns coordinates of faces
    faces = model.detect_faces(img)

    if len(faces) != 0:
        cropped_faces = []
        for face in faces:
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            face_boundary = img[y1:y2, x1:x2]
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize((244, 244))
            face_array = asarray(face_image)
            cropped_faces.append(face_array)
        #return list of cropped face images
        return cropped_faces
    else:
        print("Failed to detect face")

def get_model_scores(faces):
    samples = asarray(faces, 'float32')
    # prepare the data for the model
    samples = preprocess_input(samples, version=2)
    # create a vggface model object
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')
    # perform prediction
    return model.predict(samples)


#sorts images based on same face - returns csv of id and image files where that face exists
def write_face_dict():
    count = 1
    face_dict = {}
    scores_dict = {}
    for file in os.listdir("/Users/nisha/Downloads/Test_set/"):
        print('\n'+file)
        try:
            faces = detect_faces(file)
            if faces is not None:
                ms = get_model_scores(faces)
                for score in ms:
                    c = 0
                    for key in face_dict.keys():
                        if cosine(score, scores_dict[key]) <= 0.4:
                            face_dict[key].append(file)
                            c = 1
                    if c == 0:
                        scores_dict[count] = score
                        face_dict[count] = [file]
                        count += 1
        except ValueError:
            print("VALUE ERROR")

    with open('face_dict.csv', 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in face_dict.items():
            writer.writerow([key, value])





