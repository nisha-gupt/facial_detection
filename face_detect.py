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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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



def detect_faces(filename, model):
    filename = "/Users/nisha/Downloads/Test_set/" + filename
    #convert png to jpg
    if filename[-4:].lower() == ".png":
        img = Image.open(filename)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        os.remove(filename)
        filename = filename[:-4] + ".jpg"
        img.save(filename)

    img = pyplot.imread(filename)

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
            cropped_faces.append(face_image)
        #return list of cropped face images
        return cropped_faces
    else:
        img = Image.open(filename)
        img.save("Results/Error"+filename)
        print("Failed to detect face")

def get_model_scores(faces):
    samples = asarray(faces, 'float32')
    # prepare the data for the model
    samples = preprocess_input(samples, version=2)
    # create a vggface model object
    model = VGGFace(model='vgg16',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')
    # perform prediction
    return model.predict(samples)


#sorts images based on same face - returns csv of id and image files where that face exists
def write_face_dict():
    os.mkdir("Results/Error")
    count = 1
    scores_dict = {}
    model = MTCNN()
    for file in os.listdir("/Users/nisha/Downloads/Test_set/"):
        print('\n'+file)
        try:
            faces = detect_faces(file, model)
            if faces is not None:
                for face in faces:
                    ms = get_model_scores([asarray(face)])
                    print(ms[0])
                    c = 0
                    for key in scores_dict.keys():
                        if cosine(ms[0], scores_dict[key]) <= 0.4:
                            face.save("Results1/"+ str(key) +"/"+file)
                            c = 1
                    if c == 0:
                        scores_dict[count] = ms[0]
                        os.mkdir("Results/" + str(count))
                        face.save("Results/" + str(count) + "/" + file)
                        count += 1
        except ValueError:
            img = Image.open(file)
            img.save("Results/Error"+file)
            print("VALUE ERROR")




write_face_dict()

