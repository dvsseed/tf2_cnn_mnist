from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras import Model, datasets
import tensorflow as tf
import numpy as np
import pandas as pd
# import codecs
import os
import cv2
import gzip

do_file_preprocess = False  # 預設: 關掉下載Fashion-MNIST dataset、解壓縮及擷取轉存.jpg等處理
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.keras.backend.set_floatx('float64')
print('TensorFlow: {}'.format(tf.__version__))  # check TensorFlow version

# to get the home directory
home = str(Path.home())
home = home.replace("\\", "/")

if do_file_preprocess:
    # 下載 Fashion-MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    # 解壓.gz, ungz zip file
    def un_gz(file_name):
        # 去掉文件的名稱
        f_name = file_name.replace(".gz", "")
        # 創建gzip對象
        g_file = gzip.GzipFile(file_name)
        # gzip對象用read()打開後，寫入open()建立的文件裡
        open(f_name, "wb+").write(g_file.read())
        # 關閉gzip對象
        g_file.close()

    # 解壓縮 Fashion-MNIST dataset idx gz=>ubyte
    ufilename = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 'train-labels-idx1-ubyte', 'train-images-idx3-ubyte']
    for name in ufilename:
        ubytefile = home + '/.keras/datasets/fashion-mnist/' + name
        # 解壓.gz
        un_gz(ubytefile + '.gz')

    # 使用open()函數打開文件，並使用read()方法將所有的文件數據讀入到一個字串中
    with open(ubytefile, 'rb') as f:
        trfile = f.read()  # file是str類型，其中的每個元素是存儲的1個byte的內容

    ''' 將二進制格式的 Fashion-MNIST 數據集轉成 .jpg圖片 格式並保存，圖片標籤包含在圖片檔名中 '''

    # 將 Fashion-MNIST dataset 保存成 .jpg圖片格式: ubyte=>jpg
    def save_mnist_to_jpg(mnist_image_file, mnist_label_file, save_dir):
        if 'train' in os.path.basename(mnist_image_file):
            num_file = train_images.shape[:1][0]
            prefix = 'train'
        else:
            num_file = test_images.shape[:1][0]
            prefix = 'test'

        with open(mnist_image_file, 'rb') as f1:
            image_file = f1.read()
        with open(mnist_label_file, 'rb') as f2:
            label_file = f2.read()

        image_file = image_file[16:]
        label_file = label_file[8:]

        for i in range(num_file):
            label = label_file[i]
            image_list = [item for item in image_file[i * 784 : i * 784 + 784]]
            image_np = np.array(image_list, dtype=np.uint8).reshape(28, 28, 1)
            save_name = os.path.join(save_dir, '{}_{}_{}.jpg'.format(label, prefix, i))
            cv2.imwrite(save_name, image_np)
            print('{} ==> {}_{}_{}.jpg'.format(i, label, prefix, i))

    train_image_file = home + '/.keras/datasets/fashion-mnist/' + ufilename[3]
    train_label_file = home + '/.keras/datasets/fashion-mnist/' + ufilename[2]
    test_image_file = home + '/.keras/datasets/fashion-mnist/' + ufilename[0]
    test_label_file = home + '/.keras/datasets/fashion-mnist/' + ufilename[1]

    save_train_dir = home + '/.keras/datasets/fashion-mnist/train_images/'
    save_test_dir = home + '/.keras/datasets/fashion-mnist/test_images/'

    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    save_mnist_to_jpg(train_image_file, train_label_file, save_train_dir)
    save_mnist_to_jpg(test_image_file, test_label_file, save_test_dir)


# Loading Training data
train_images = []
train_labels = []

famnist = home + '/.keras/datasets/fashion-mnist/'
fatrmnist = famnist + 'train_images'
for img_path in os.listdir(fatrmnist):
    im = Image.open(fatrmnist + '/' + str(img_path))
    # im = im.resize((100, 100))
    train_images.append(np.array(im))
    train_labels.append(img_path[:1])

train_images = np.array(train_images)
train_images = np.expand_dims(train_images, axis=-1)
train_labels = np.array(train_labels)
# print(train_images.shape)
# print(train_labels.shape)

train_images = train_images / 255.0  # Image Normalization

# reshape labels of training data
train_labels = np.reshape(train_labels, (-1, 1))
# One-hot encoding training labels
enc = OneHotEncoder(categories='auto')
train_labels = enc.fit_transform(train_labels).toarray()

# print(train_images.shape)
# print(train_labels.shape)

# to assign the class
num_classes = int(train_labels.shape[1:2][0])

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).repeat(1).batch(32)

# Building model
# Model Architecture
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            name='conv1')
        # self.bn1 = BatchNormalization(axis=-1, name='bn1')
        self.pool1 = MaxPool2D(pool_size=(2, 2), name='maxpool1')
        self.conv2 = Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            name='conv2')
        # self.bn2 = BatchNormalization(axis=-1, name='bn2')
        self.pool2 = MaxPool2D(pool_size=(2, 2), name='maxpool2')
        self.flatten = Flatten()
        self.d1 = Dense(units=512,
                        activation='relu',
                        name='fc1')
        # self.dropout1 = Dropout(rate=0.4, name='dropout1')
        self.d2 = Dense(units=128,
                        activation='relu',
                        name='fc2')
        self.dropout2 = Dropout(rate=0.4, name='dropout2')
        self.d3 = Dense(units=num_classes,
                        activation='softmax',
                        name='output')

    def call(self, x, is_training=False):
        x = self.conv1(x)
        # x = self.bn1(x, training=is_training)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = self.bn2(x, training=is_training)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        # x = self.dropout1(x)
        x = self.d2(x)
        x = self.dropout2(x)
        x = self.d3(x)
        return x


model = MyModel()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, is_training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# model training
EPOCHS = 10
print('>>CNN Model Training...>>')
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        # print(images.shape, labels.shape)
        train_step(images, labels)

    model.save_weights('./content', save_format='tf')
    print('Epoch: {:2},  Loss:{:7.4f},  Accuracy: {:7.4f}'.format(epoch + 1,
                                                                  train_loss.result(),
                                                                  train_accuracy.result() * 100))

    train_loss.reset_states()
    train_accuracy.reset_states()


print('\n--CNN Model architecture--')
print(model.summary())
print('Layers: {}'.format(len(model.layers)))

# Loading Testing data
test_images = []
test_labels = []
fatemnist = famnist + '/test_images'
for img_path in os.listdir(fatemnist):
    im = Image.open(fatemnist + '/' + str(img_path))
    # im = im.resize((100, 100))
    test_images.append(np.array(im))
    test_labels.append(img_path[:1])

test_images = np.array(test_images)
test_labels = np.array(test_labels)
# print(test_images.shape)
# print(test_labels.shape)

# Image Normalization
test_images = test_images / 255.0

# reshape labels of training data
test_labels = np.reshape(test_labels, (-1, 1))
# One-hot encoding training labels
enc = OneHotEncoder(categories='auto')
test_labels = enc.fit_transform(test_labels).toarray()

model.load_weights('./content')

# Predicting on Test Set
predictions = []
print('>>CNN Model predicting...>>')
for img in test_images:
    # img = img.reshape(1, 100, 100, 3)
    img = img.reshape((1,) + train_images.shape[1:])
    predictions.append(np.argmax(model(img, is_training=False), axis=1))

predictions = np.array(predictions)
# print(predictions.shape)

df = pd.DataFrame(predictions)
# print(df.shape)
# print(df.columns)
# print(df.describe())
print('>>Save the csv file...>>')
df.to_csv('./content/pred.csv')

# 秀出 predict與label(y) 不相符的項目
# print('index  predict<>label')
# for i in range(len(df)):
#     if df[0][i] != np.argmax(test_labels[i]):
#         print(' {:4}      {:2} !={:2}'.format(i, df[0][i], np.argmax(test_labels[i])))

# 讀取 CSV File
# rdf = pd.read_csv('./content/pred.csv')
# print(rdf)
print('**All done.**')
