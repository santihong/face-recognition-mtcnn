# develop a classifier for the 5 Celebrity Faces Dataset
import cv2 as cv
import time

from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model


BASE_DIR = 'res/'


# load face embeddings
data = load(BASE_DIR + '5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# load the facenet model
model_embedding = load_model(BASE_DIR + 'facenet_keras.h5')
print('Loaded Model')
model_embedding.summary()


def get_embedding_multi(model, faces):
    faces = faces.astype('float32')

    # normalize
    for face in faces:
        mean, std = face.mean(), face.std()
        face -= mean
        face /= std

    predictions = model.predict(faces)
    return predictions


def extract_face(image, required_size=(160, 160)):
    '''
    Return face image and face info
    :param image:
    :param required_size:
    :return:
    '''
    pixels = asarray(image)

    detector = MTCNN(min_face_size=100)
    results = detector.detect_faces(pixels)

    for info in results:
        print(info)
        x1, y1, width, height = info['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        info['face'] = face_array
    return results


def detect(image):
    faces_info = extract_face(image)
    if len(faces_info) == 0:
        return None

    faces = []
    # draw face bounding box and keypoints
    for face_info in faces_info:
        faces.append(face_info['face'])

    # get face image embedding from facenet model
    embeddings = get_embedding_multi(model_embedding, asarray(faces))

    # predict
    # samples = expand_dims(embeddings, axis=0)
    predict_classes = model.predict(embeddings)
    predict_probs = model.predict_proba(embeddings)
    predict_names = out_encoder.inverse_transform(asarray(predict_classes))

    # draw bouding box and keypoints
    for ind in range(len(faces_info)):
        face_info = faces_info[ind]

        class_index = predict_classes[ind]
        class_probability = predict_probs[ind, class_index] * 100
        predict_name = predict_names[ind]

        if class_probability < 60:
            continue

        x1, y1, width, height = face_info['box']
        x2, y2 = x1 + width, y1 + height
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        keypoints = face_info['keypoints']
        for key, point in keypoints.items():
            cv.circle(image, point, 5, (0, 0, 255), thickness=-1)

        cv.putText(image, '{}({}%)'.format(predict_name, int(class_probability)), (int((x1+x2)/2), y1),
                   cv.QT_FONT_NORMAL, 1, (0, 255, 0), thickness=1)

    return image


def take_video():
    cap = cv.VideoCapture(0)
    # todo Set frame size not working
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        print('Error, camera cant open')
        exit(-1)

    wind_name = "video"
    cv.namedWindow(wind_name, flags=cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
    time_last_frame = 0

    cur_frame = 0
    per_frames = 30
    while True:
        ret, frame = cap.read()
        cur_frame += 1

        if not ret:
            print('Capture frame failed')
            exit(-1)

        image = frame

        # perform face detection
        detect_image = None
        if cur_frame % per_frames == 0:
            detect_image = detect(image)
            if detect_image is not None:
                image = detect_image

        now = time.time()
        time_elapsed = now - time_last_frame
        time_last_frame = now
        frame_rate = int(1 / time_elapsed)
        cv.putText(image, 'FPS: %d' % frame_rate, (25, 100), cv.QT_FONT_NORMAL, 2, (255, 0, 0), thickness=2)
        cv.imshow(wind_name, image)

        wait = 1 if detect_image is None else 0
        c = cv.waitKey(wait)

        if c == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


take_video()

