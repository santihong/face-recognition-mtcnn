# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot as plt

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


def predict():
    # load faces
    data = load(BASE_DIR + '5-celebrity-faces-dataset.npz')
    test_imgs = data['arr_2']
    test_labels = testy
    test_embeddings = testX

    predict_classes = model.predict(test_embeddings)
    predict_probs = model.predict_proba(test_embeddings)

    # Show prediction
    num_rows = 5
    num_cols = 5
    num_images = min(num_rows * num_cols, len(test_imgs))
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        class_index = predict_classes[i]
        plot_image(test_imgs[i], class_index, predict_probs[i][class_index], test_labels[i])
    plt.show()


def plot_image(img, predict_class, predict_prob, true_label):
    predict_name = out_encoder.inverse_transform([predict_class])[0]
    true_name = out_encoder.inverse_transform([true_label])[0]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if predict_class == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}\n {:2.0f}% ({})".format(predict_name,
                                         100*predict_prob,
                                         true_name),
               color=color)


predict()

