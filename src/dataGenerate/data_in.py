import cv2
import random
import tensorflow as tf
import sys
sys.path.append('../../pub/')
from file import *


# very 'r_size' a record
def encode(data_root, record_path, r_size=1000):
    names = get_file_name(data_root)
    # print(len(names))
    num_example = 0
    while len(names) > 500:
        if not num_example % r_size:
            writer = tf.python_io.TFRecordWriter(record_path + '%s-%s.tfrecords' %(num_example, num_example+r_size))
        name = random.choice(names)
        names.remove(name)
        image = cv2.imread(data_root+name, 0)
        # image = cv2.resize(image, (100, 100))
        if image is None:
            print(num_example, name)
            continue
        # print(image.shape)
        # print(height, width, nchannel)
        label = int(name[0])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        serialized = example.SerializeToString()
        writer.write(serialized)
        num_example += 1
        if not num_example % r_size:
            writer.close()
        if not num_example % 100:
            print("Sample Data Cnt:", num_example)


# decode to tensor
def decode_from_tfrecords(filename, image_size, num_epoch=None,):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=num_epoch)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    label = tf.cast(example['label'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    # image = tf.reshape(image, tf.stack([
    #     tf.cast(example['height'], tf.int32),
    #     tf.cast(example['width'], tf.int32),
    #     tf.cast(example['nchannel'], tf.int32)]))

    # above 3 lines code equals to underline
    image = tf.reshape(image, image_size)
    return image, label


def get_batch(images, labels, crop_size, batch_size=10, ):
    ims = tf.random_crop(images, crop_size)
    image, label = tf.train.shuffle_batch(
        [ims, labels],
        batch_size=batch_size,
        capacity=1000 + 3*batch_size,
        min_after_dequeue=50
    )
    return image, tf.one_hot(label, 2)


def test():
    encode('F:/DL/data/vpr_data/train/', './')
    image, label = decode_from_tfrecords('./0-100.tfrecords')
    batch_image, batch_label = get_batch(image, label, batch_size=50)  # batch:10
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        # session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for l in range(1):
            batch_image_np, batch_label_np = session.run([batch_image, batch_label])

            print(len(batch_image_np), len(batch_label_np))
            print(batch_label_np)
        coord.request_stop()  # queue should be closed
        coord.join(threads)

if __name__ == '__main__':
    # test()
    # image, label = decode_from_tfrecords('./0-100.tfrecords')
    # print(tf.shape(image))
    encode('F:/DL/data/vpr_data/train/', '../../data/train/', r_size=10000)
    pass