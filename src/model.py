import tensorflow as tf
import cv2
import os
import sys
sys.path.append('../pub/')
from data_in import *


if 0:
#   400*50 image
    with tf.variable_scope("weights"):
        weights = {
            # 400*50*3->350*40*30->175*20*30
            'conv1': tf.get_variable('conv1', [51, 11, 3, 30], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 175*20*30->90*18*50->45*9*50
            'conv2': tf.get_variable('conv2', [86, 3, 30, 50], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 45*9*50->42*6*60->21*3*60
            'conv3': tf.get_variable('conv3', [4, 4, 50, 60], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 21*3*60->120
            'fc1': tf.get_variable('fc1', [21 * 3 * 60, 120], initializer=tf.contrib.layers.xavier_initializer()),
            # 120->6
            'fc2': tf.get_variable('fc2', [120, 2], initializer=tf.contrib.layers.xavier_initializer()),
        }
    with tf.variable_scope('bias'):
        biases = {
                'conv1': tf.get_variable('conv1', [30, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [50, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [60, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [120, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [2, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

if 0:# 显存不够，报错
    crop_scale = 400
    #   400*400 iamge
    with tf.variable_scope("weights"):
        weights = {
            # 400*400*3->350*350*30->175*175*30
            'conv1': tf.get_variable('conv1', [51, 51, 3, 30], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 175*175*30->126*126*50->63*63*50
            'conv2': tf.get_variable('conv2', [50, 50, 30, 50], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 63*63*50->56*56*60->28*28*60
            'conv3': tf.get_variable('conv3', [8, 8, 50, 60], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 28*28*60->120
            'fc1': tf.get_variable('fc1', [28 * 28 * 60, 120], initializer=tf.contrib.layers.xavier_initializer()),
            # 120->6
            'fc2': tf.get_variable('fc2', [120, 2], initializer=tf.contrib.layers.xavier_initializer()),
        }
    with tf.variable_scope('bias'):
        biases = {
                'conv1': tf.get_variable('conv1', [30, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [50, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [60, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [120, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [2, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }


if 0:# 还是TMD的大了，有时候可以运行
    crop_scale = 200
    #   200*200 iamge
    with tf.variable_scope("weights"):
        weights = {
            # 200*200*3->150*150*30->75*75*30
            'conv1': tf.get_variable('conv1', [51, 51, 3, 30], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 75*75*30->66*66*50->33*33*50
            'conv2': tf.get_variable('conv2', [10, 10, 30, 50], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 33*33*50->26*26*60->13*13*60
            'conv3': tf.get_variable('conv3', [8, 8, 50, 60], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 13*13*60->120
            'fc1': tf.get_variable('fc1', [13 * 13 * 60, 120], initializer=tf.contrib.layers.xavier_initializer()),
            # 120->6
            'fc2': tf.get_variable('fc2', [120, 2], initializer=tf.contrib.layers.xavier_initializer()),
        }
    with tf.variable_scope('bias'):
        biases = {
                'conv1': tf.get_variable('conv1', [30, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [50, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [60, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [120, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [2, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

if 1:
    crop_scale = 100
    #   100*100 iamge
    with tf.variable_scope("weights"):
        weights = {
            # 100*100*3->70*70*30->35*35*30
            'conv1': tf.get_variable('conv1', [31, 31, 3, 30], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 35*35*30->28*28*50->14*14*50
            'conv2': tf.get_variable('conv2', [8, 8, 30, 50], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 14*14*50->10*10*60->5*5*60
            'conv3': tf.get_variable('conv3', [5, 5, 50, 60], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            # 5*5*60->120
            'fc1': tf.get_variable('fc1', [5 * 5 * 60, 120], initializer=tf.contrib.layers.xavier_initializer()),
            # 120->6
            'fc2': tf.get_variable('fc2', [120, 2], initializer=tf.contrib.layers.xavier_initializer()),
        }
    with tf.variable_scope('bias'):
        biases = {
                'conv1': tf.get_variable('conv1', [30, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [50, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [60, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [120, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [2, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }
        


with tf.device('/gpu:0'):
    def inference(images, keep_prob=0.5):
        # 向量转为矩阵
        #images = tf.reshape(images, shape=[-1, crop_scale, crop_scale, 3])  # [batch, in_height, in_width, in_channels]
        images = tf.reshape(images, shape=[-1, crop_scale, crop_scale, 3])  # [batch, in_height, in_width, in_channels]
        images = tf.cast(images, tf.float32) / 255. # 归一化处理

        # 第一层
        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                               biases['conv1'])

        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二层
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                               biases['conv2'])
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第三层
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                               biases['conv3'])
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, weights['fc1'].get_shape().as_list()[0]])

        drop1 = tf.nn.dropout(flatten, keep_prob)
        fc1 = tf.matmul(drop1, weights['fc1']) + biases['fc1']

        fc_relu1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc_relu1, weights['fc2']) + biases['fc2']
        return fc2
'''
with tf.device('/gpu:0'):
    def inference_test(images):
        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 400, 400, 3])  # [batch, in_height, in_width, in_channels]
        images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2  # 归一化处理

        # 第一层
        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                                   biases['conv1'])

        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二层
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                                   biases['conv2'])
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第三层
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                                   biases['conv3'])
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, weights['fc1'].get_shape().as_list()[0]])

        fc1 = tf.matmul(flatten, weights['fc1']) + biases['fc1']
        fc_relu1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc_relu1, weights['fc2']) + biases['fc2']
        return fc2
'''

def softmax_loss(predicts, labels):
    predicts = tf.nn.softmax(predicts)
    with tf.device('/cpu:0'):
        labels = tf.one_hot(labels, weights['fc2'].get_shape().as_list()[1])
    loss = -tf.reduce_mean(labels * tf.log(predicts))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
    return loss


def optimer(loss, lr=0.000001):
    train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    return train_optimizer


def train():
    encode_to_tfrecords("../data/train/data.txt", "../data/train/spec/", 'train.tfrecords')
    image, label = decode_from_tfrecords('../data/train/spec/train.tfrecords')
    batch_image, batch_label = get_batch(image,label,batch_size=60,crop_size=100)
    #batch_image, batch_label = get_batch(image,label,batch_size=10,crop_size=39)

    inf = inference(batch_image,keep_prob=0.5)
    loss = softmax_loss(inf, batch_label)
    opti = optimer(loss)

    # validation
    encode_to_tfrecords("../data/validate/data.txt", "../data/validate/", 'val.tfrecords')
    test_image, test_label = decode_from_tfrecords('../data/validate/val.tfrecords', num_epoch=None)
    test_images, test_labels = get_test_batch(test_image, test_label, batch_size=50, crop_size=crop_scale)  # batch 生成测试
    with tf.device('/cpu:0'):
        test_onehot_labels = tf.one_hot(test_labels, weights['fc2'].get_shape().as_list()[1])
    test_inf = inference(test_images, 1.0)

    correct_prediction = tf.equal(tf.argmax(test_inf, 1), tf.argmax(test_onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.scalar_summary('Accuarcy', accuracy)
    # tf.scalar_summary('Loss', loss)
    # tf.image_summary('Train_Image',image)
    # merged_summary_op = tf.merge_all_summaries()

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        #summary_writer = tf.train.SummaryWriter('E:/PycharmProjects/Voice/guitar_piano/', graph_def=session.graph_def)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        max_iter = 2
        iter = 0
        if os.path.exists(os.path.join("../models/", 'model.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join("../models/", 'model.ckpt'))
        while iter < max_iter:
            #loss_np, _, label_np, image_np, inf_np = session.run([loss, opti, batch_label, batch_image, inf])
            print(session.run(loss), session.run(accuracy))
            # print image_np.shape
            #cv2.imshow(str(label_np[0]),image_np[0])
            # print label_np[0]
            # cv2.waitKey()
            # print label_np
            #if iter % 10 == 0:
                #print('trainloss:', loss_np)
            #if iter % 50 == 0:
            #    accuracy_np,summary_str = session.run([accuracy,merged_summary_op])
            #    print('test accruacy:', accuracy_np)
            #    summary_writer.add_summary(summary_str,iter)
                #tf.train.Saver(max_to_keep=None).save(session, os.path.join('model', 'model.ckpt'))
            iter += 1
        tf.train.Saver(max_to_keep=None).save(session, os.path.join('../models/', 'model.ckpt'))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()


