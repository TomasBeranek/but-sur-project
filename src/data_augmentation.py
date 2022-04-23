import sys
import matplotlib.pyplot as plt
from ikrlib import png2fea
import tensorflow as tf
import tensorflow_addons as tfa
import math
from random import randint, sample
import numpy as np
import shutil

def visualize(original, augmented_images):
    for i in range(len(augmented_images)):
        plt.subplot(4,5,i+2)
        plt.imshow(augmented_images[i], cmap='gray')
        plt.axis('Off')

    plt.subplot(4,5,1)
    plt.title('Original image')
    plt.imshow(original, cmap='gray')
    plt.axis('Off')

    plt.show()

def augment_single_image(original, ops):
    augmented_images = []

    # generate len(ops) augmented images from given image
    for output_image_ops in ops:
        augmented = original

        # apply transformations
        for operation in output_image_ops:
            # mirror
            if operation == 0:
                augmented = tf.image.flip_left_right(augmented)
            # random brightness
            elif operation == 1:
                augmented = tf.image.random_brightness(augmented, max_delta=70)
                augmented = tf.clip_by_value(augmented, 0.0, 255.0)
            # random rotation
            elif operation == 2:
                augmented = tfa.image.rotate(augmented, randint(0,5) * math.pi / 180, interpolation='bilinear')
                augmented = tf.image.central_crop(augmented, 0.95)
                augmented = tf.image.resize(augmented, (80,80))
            # random shift
            elif operation == 3:
                augmented = tf.keras.preprocessing.image.random_shift(augmented, 0.15, 0.15, fill_mode='nearest', interpolation_order=1)
            # random zoom
            elif operation == 4:
                augmented = tf.keras.preprocessing.image.random_zoom(augmented, (0.95,1.05), fill_mode='nearest', interpolation_order=1)
            # random noise
            elif operation == 5:
                noise = tf.random.normal(shape=tf.shape(augmented), mean=0.0, stddev=4)
                augmented = tf.add(augmented, noise)
                augmented = tf.clip_by_value(augmented, 0.0, 255.0)

        augmented_images += [augmented]

    #visualize(original, augmented_images)
    return augmented_images

def augment_dir(src_dir, dest_dir, reproduce_coef):
    images = png2fea(src_dir, verbose=False)

    for file, original in images.items():
        # add channels -- sinice its a greyscale add only 1 channel
        original = original.reshape(80,80,1)

        # convert numpy array to tensorflow tensor
        original = tf.image.resize(original, (80, 80))

        ops = []

        # randomly choose augmentation operations
        while len(ops) < (reproduce_coef-1):
            # maximum number of operations is 5 since we cant use the same operation twice
            # because it might generate too much distorted images -- e.g. 5x noise
            single_ops_len = randint(1,5)
            new_ops_variant = sample(range(0,5), single_ops_len)

            # we dont want only mirror twice, since it would generate same augmented images
            if new_ops_variant == [0] and new_ops_variant in ops:
                continue

            ops += [new_ops_variant]

        # generate 'reproduce_coef' augmented images from original image
        augmented_images = augment_single_image(original, ops)

        filename = file.split('/')[-1].split('.')[0]

        # safe original image
        shutil.copy(file, dest_dir + "/" + filename + "_aug0.png")

        # safe augmented images
        cnt = 1
        for augmented_image in augmented_images:
            augmented_image = tf.repeat(augmented_image, repeats=[3], axis=2)
            # convert it from tensor to numpy array because imsave cant read tf.float64 type
            augmented_image = augmented_image.numpy()
            augmented_image = augmented_image/255
            plt.imsave(dest_dir + "/" + filename + ("_aug%d.png" % cnt), augmented_image)
            cnt += 1

if __name__ == '__main__':
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    reproduce_coef = int(sys.argv[3])
    augment_dir(src_dir, dest_dir, reproduce_coef)
