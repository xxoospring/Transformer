import cv2
import tensorflow as tf
import sys
sys.path.append('../pub/')
from file import *


# encode to tfrecordes
def encode_to_tfrecords(label_file, data_root, new_name='data.tfrecords', resize=None):
    if file_exist(data_root, new_name):
        return
    writer = tf.python_io.TFRecordWriter(data_root + '/' + new_name)
    num_example = 0
    with open(label_file, 'r') as f:
        for l in f.readlines():
            l = l.split()
            # print(l[0])
            image = cv2.imread(data_root + "/" + l[0])
            if resize is not None:
                image = cv2.resize(image, resize)
            height, width, nchannel = image.shape

            label = int(l[1])

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            num_example += 1
    print(label_file, "Sample Data Cnt:", num_example)
    writer.close()


# decode to tensor
def decode_from_tfrecords(filename, num_epoch=None):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=num_epoch)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'nchannel': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    label = tf.cast(example['label'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, tf.stack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))
    # label=example['label']
    return image, label
# images, labels = decode_from_tfrecords('../data/train/data.tfrecords')


def get_batch(image, label, batch_size, crop_size=None):
    # function like 'const'
    distorted_image = image
    if crop_size is not None:
        distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])  # random crop
    distorted_image = tf.image.random_flip_up_down(distorted_image)  # random flip in vertical direction
    # distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#brightness
    # distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#contrast

    # generate batch
    # shuffle_batch：capacity defines range of shuffle
    # capacity is the scale to shuffle batch_size images
    #images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size,
    #                                             num_threads=1, capacity=200, min_after_dequeue=50)
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size,
                                                 num_threads=1, capacity=10000, min_after_dequeue=500)
    # images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)
    # tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def get_none_crop_batch(image, label, batch_size):
    distorted_image = tf.image.random_flip_up_down(image)
    images, label_batch = tf.train.shuffle_batch_join([distorted_image, label], batch_size=batch_size,
                                                 capacity=150, min_after_dequeue=30)
    # tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def get_test_batch(image, label, batch_size, crop_size):
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size,capacity=150, min_after_dequeue=30)
    return images, tf.reshape(label_batch, [batch_size])


def test():
    encode_to_tfrecords("../data/train/data.txt", "../data/train")
    image, label = decode_from_tfrecords('../data/train/spec/data.tfrecords')
    batch_image, batch_label = get_batch(image, label, 10, 400)  # batch:10,crop:400x400
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for l in range(1):
            # image_np,label_np=session.run([image,label])
            # cv2.imshow("temp",image_np)
            # cv2.waitKey()
            # print label_np
            # print image_np.shape

            batch_image_np, batch_label_np = session.run([batch_image, batch_label])
            print(type(batch_image_np))
            cv2.imshow("temp",batch_image_np[0])
            cv2.waitKey()
        coord.request_stop()  # queue should be closed
        coord.join(threads)
#test()    #   test OK!

if __name__ == '__main__':
    # encode_to_tfrecords('../data/train/data.txt', '../data/train/spec/')  # test OK!
    # images, labels = decode_from_tfrecords('../data/train/spec/data.tfrecords')
    # print(type(images),type(labels))
    test()