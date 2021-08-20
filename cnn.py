import pandas as pd
import numpy as np
import cv2
import os

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from os.path import isfile, join
from keras import backend as K
from os import listdir
from PIL import Image

index_by_directory = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '+': 10,
    '-': 11,
    'x': 12
}


def get_index_by_directory(directory):
    return index_by_directory[directory]


# Code for training data
def load_images_from_folder(folder):
    train_data = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # Convert to Image to Grayscale
        img = ~img  # Invert the bits of image 255 -> 0
        if img is not None:
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Set bits > 127 to 1 and <= 127 to 0
            ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])  # Sort by x
            maxi = 0
            for c in cnt:
                x, y, w, h = cv2.boundingRect(c)
                maxi = max(w * h, maxi)
                if maxi == w * h:
                    x_max = x
                    y_max = y
                    w_max = w
                    h_max = h
            im_crop = thresh[y_max:y_max + h_max + 10, x_max:x_max + w_max + 10]  # Crop the image as most as possible
            im_resize = cv2.resize(im_crop, (28, 28))  # Resize to (28, 28)
            im_resize = np.reshape(im_resize, (784, 1))  # Flat the matrix
            train_data.append(im_resize)
    return train_data


# This code generates CSV for training data
def load_all_imgs():
    dataset_dir = "./datasets/"  # Read dataset directory
    directory_list = listdir(dataset_dir)
    first = True
    data = []

    print('Exporting images...')
    for directory in directory_list:  # loop through each digit folder in dataset directory
        print(directory)
        if first:
            first = False
            data = load_images_from_folder(dataset_dir + directory)
            for i in range(0, len(data)):
                data[i] = np.append(data[i], [str(get_index_by_directory(directory))])
            continue

        aux_data = load_images_from_folder(dataset_dir + directory)
        for i in range(0, len(aux_data)):
            aux_data[i] = np.append(aux_data[i], [str(get_index_by_directory(directory))])
        data = np.concatenate((data, aux_data))

    df = pd.DataFrame(data, index=None)
    df.to_csv('model/train_data.csv', index=False)


def extract_imgs(img):
    img = ~img  # Invert the bits of image 255 -> 0
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Set bits > 127 to 1 and <= 127 to 0
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])  # Sort by x

    img_data = []
    rects = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        rect = [x, y, w, h]
        rects.append(rect)

    bool_rect = []
    # Check when two rectangles collide
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec != r:
                if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and \
                        rec[1] < (r[1] + r[3] + 10):
                    flag = 1
                l.append(flag)
            else:
                l.append(0)
        bool_rect.append(l)

    dump_rect = []
    # Discard the small collide rectangle
    for i in range(0, len(cnt)):
        for j in range(0, len(cnt)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if area1 == min(area1, area2):
                    dump_rect.append(rects[i])

    # Get the final rectangles
    final_rect = [i for i in rects if i not in dump_rect]
    for r in final_rect:
        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]

        im_crop = thresh[y:y + h + 10, x:x + w + 10]  # Crop the image as most as possible
        im_resize = cv2.resize(im_crop, (28, 28))  # Resize to (28, 28)
        im_resize = np.reshape(im_resize, (1, 28, 28))  # Flat the matrix
        img_data.append(im_resize)

    return img_data


# Model creation using convolutional neural network
class ConvolutionalNeuralNetwork:
    def __init__(self):
        if os.path.exists('model/model_weights.h5') and os.path.exists('model/model.json'):
            self.load_model()
        else:
            self.create_model()
            self.train_model()
            self.export_model()

    def create_model(self):
        first_conv_num_filters = 30
        first_conv_filter_size = 5
        second_conv_num_filters = 15
        second_conv_filter_size = 3
        pool_size = 2

        # Create model
        print("Creating Model...")
        self.model = Sequential()
        self.model.add(
            Conv2D(first_conv_num_filters, first_conv_filter_size, input_shape=(28, 28, 1), activation='relu'))

        # 1. Convolution Operation :
        # In this process, we reduce the size of the image by passing the input image through a Feature detector/Filter/Kernel 
        # so as to convert it into a Feature Map/ Convolved feature/ Activation Map
        # It helps remove the unnecessary details from the image.
        # We can create many feature maps (detects certain features from the image) to obtain our first convolution layer.
        # Involves element-wise multiplication of convolutional filter with the slice of an input matrix and finally the 
        # summation of all values in the resulting matrix.

        self.model.add(MaxPooling2D(pool_size=pool_size))
        # It helps to reduce the spatial size of the convolved feature which in-turn helps to to decrease the 
        # computational power required to process the data.
        # Here we are able to preserve the dominant features, thus helping in the process of effectively training the model.
        # Converts the Feature Map into a Pooled Feature Map.
        # Pooling is divided into 2 types:
        # 1. Max Pooling - Returns the max value from the portion of the image covered by the kernel.
        # 2. Average Pooling - Returns the average of all values from the portion of the image covered by the kernel.
        self.model.add(Conv2D(second_conv_num_filters, second_conv_filter_size, activation='relu'))

        # 1.2. ReLU Activation Function :
        # ReLU is the most commonly used activation function in the world.
        # When applying convolution, there is a risk we might create something linear and there we need to break linearity.
        # Rectified Linear unit can be described by the function f(x) = max(x, 0).
        # We are applying the rectifier to increase the non-linearity in our image/CNN. Rectifier keeps only non-negative values
        #  of an image.

        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(0.2))
        # Usually, when all the features are connected to the FC layer, it can cause overfitting in the training dataset.
        # Overfitting occurs when a particular model works so well on the training data causing a negative impact in the 
        # model’s performance when used on a new data.

        # To overcome this problem, a dropout layer is utilised wherein a few neurons are dropped from the neural network 
        # during training process resulting in reduced size of the model. On passing a dropout of 0.3, 30% of the nodes are 
        # dropped out randomly from the neural network.
        self.model.add(Flatten())
        #  there are two Convolution and Pooling pairs followed by a Flatten layer which is usually used as a connection 
        # between Convolution and the Dense layers.

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(13, activation='softmax'))
        # softmax functions are preferred an for a multi-class classification, generally softmax us used.

        # Compile the model
        print("Compiling Model...")
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def load_model(self):
        print('Loading Model...')
        model_json = open('model/model.json', 'r')
        loaded_model_json = model_json.read()
        model_json.close()
        loaded_model = model_from_json(loaded_model_json)

        print('Loading weights...')
        loaded_model.load_weights("model/model_weights.h5")

        self.model = loaded_model

    def train_model(self):
        if not os.path.exists('model/train_data.csv'):
            load_all_imgs()

        csv_train_data = pd.read_csv('model/train_data.csv', index_col=False)

        # The last column contain the results
        y_train = csv_train_data[['784']]
        csv_train_data.drop(csv_train_data.columns[[784]], axis=1, inplace=True)
        csv_train_data.head()

        y_train = np.array(y_train)

        x_train = []
        for i in range(len(csv_train_data)):
            x_train.append(np.array(csv_train_data[i:i + 1]).reshape(1, 28, 28))
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (-1, 28, 28, 1))

        # Train the model.
        print('Training model...')
        self.model.fit(
            x_train,
            to_categorical(y_train, num_classes=13),
            epochs=10,
            batch_size=200,
            shuffle=True,
            verbose=1
        )

    def export_model(self):
        model_json = self.model.to_json()
        with open('model/model.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('model/model_weights.h5')

    def predict(self, operationBytes):
        Image.open(operationBytes).save('aux.png')
        img = cv2.imread('aux.png', cv2.IMREAD_GRAYSCALE)
        os.remove('aux.png')

        if img is not None:
            img_data = extract_imgs(img)

            operation = ''
            for i in range(len(img_data)):
                img_data[i] = np.array(img_data[i])
                img_data[i] = img_data[i].reshape(-1, 28, 28, 1)

                result = self.model.predict_classes(img_data[i])

                if result[0] == 10:
                    operation += '+'
                elif result[0] == 11:
                    operation += '-'
                elif result[0] == 12:
                    operation += 'x'
                else:
                    operation += str(result[0])

            return operation


augmentation