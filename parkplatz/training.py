#!/usr/bin/env python3

import tensorflow as tf


def main():
    sess = tf.Session()

    DEVICE_GPU = "GPU"
    DEVICE_CPU = "/cpu:0"

    with tf.device(DEVICE_GPU):
      from cars import cars_data
      d = cars_data(batch_size=64, sess=sess)
      image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()
    image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
    print(image_batch.shape)
    print(target_batch.shape)


if __name__ == '__main__':
    main()
