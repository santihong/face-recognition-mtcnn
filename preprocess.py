from os import listdir
from os.path import isdir

from PIL import Image
from numpy import asarray
from numpy import savez_compressed
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

BASE_DIR = 'res/'


def extract_face(filename, required_size=(160, 160)):
    """
    Extract one face use mtcnn from image
    :param filename:
    :param required_size:
    :return:
    """
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# specify folder to plot
folder = BASE_DIR + '5-celebrity-faces-dataset/train/ben_afflek/'
i = 1
# enumerate files
for filename in listdir(folder):
    path = folder + filename
    face = extract_face(path)
    print(i, face.shape)

    # plot
    pyplot.subplot(2, 7, i)
    pyplot.axis('off')
    pyplot.imshow(face)
    i += 1

pyplot.show()


# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue

        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]

        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# load train dataset
trainX, trainy = load_dataset(BASE_DIR + '5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)

# load test dataset
testX, testy = load_dataset(BASE_DIR + '5-celebrity-faces-dataset/val/')
print(testX.shape, testy.shape)

# save arrays to one file in compressed format
savez_compressed(BASE_DIR + '5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
