# An undercomplete autoencoder on MNIST dataset
# https://github.com/Machinelearninguru/Deep_Learning/blob/master/TensorFlow/neural_networks/autoencoder/simple_autoencoder.py
import tensorflow.contrib.layers as lays
import os
import tensorflow as tf
import pandas
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from image_reader import create_map_file

folder = "101_ObjectCategories"    
batch_size = 5  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate
rsize = (32,32)
batch_per_ep = 1




def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 3))
    resized_imgs = np.zeros((imgs.shape[0], rsize[0], rsize[1], 3))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (rsize[0], rsize[1]))
    return resized_imgs


def autoencoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    # label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpg(file_contents, channels=3)
    return example

# create a file with image names
map_file = os.path.join(os.path.dirname(__file__), folder, "map_file.txt")
if not os.path.exists(map_file):
    create_map_file(map_file, folder, "jpg", class_mapping=None, include_unknown=True)
    
# create a list of image naes
dfimages = pandas.read_csv(map_file, sep="\t", header=None)
dfimages.columns = ["image", "label"]
print(dfimages.head(n=2))
image_list = list(dfimages["image"])

# queue
filename_queue = tf.train.string_input_producer(image_list)
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_jpeg(content, channels=3)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [rsize[0], rsize[1]])

# batch
image_batch_train = tf.train.batch([resized_image], batch_size=batch_size)
help(image_batch_train)

# inputs
ae_inputs = tf.placeholder(tf.float32, (None, rsize[0], rsize[1], 3)) 
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: image_batch_train})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]

    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
plt.show()