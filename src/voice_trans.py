from pylab import *
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy import *
import cv2
import os
from public import *


# transform wav file into specgram image
def trans2spec(wav, store_path, label):
    '''
    :param wav: 源音频信号
    :param store_path
    :param label 
    :return: label
    1.首先依据声强的差值关系简单确立起止点
    2.再将每个crop转换成对应的语谱图
    注意点：差值是根据经验确定的，不同的声纹是否能通用？
    '''
    # get voice sample frequency and data
    sam_freq, snd = wavfile.read(wav)
    #print(sam_freq)
    sam_num, nchannels = shape(snd)
    # use one channel
    s1 = snd[:, 0]
    len_wav = len(s1)
    slot = 500  # crop wav into 500ms pieces
    pace = int(float(slot) * sam_freq / 1000.0)

    cnt_lst = []
    for index in arange(0, len_wav, pace):
        chuck = s1[index:index+pace]
        if sum(abs(chuck)) < 1000000:
            continue
        cnt_lst.append(int(float(index) / pace))
        plt.specgram(chuck, Fs=sam_freq, scale_by_freq=True, sides='default')
        plt.savefig(store_path + str(int(float(index)/pace)) + '_' + str(label) + '.jpg')
    return cnt_lst


def row_edge_detect(img_path):
    top_line = 0
    bottom_line = 0
    img = cv2.imread(img_path)
    row = img.shape[0]
    clo = img.shape[1]
    for r in range(row):
        avg = mean(img[r])
        if avg > 245 and avg < 250:
            top_line = r
        if avg>=0 and avg < 82:
            bottom_line = r
    if False:# debug code
        print(top_line, bottom_line)
        # print(avg)
        img = cv2.imread('test.jpg')
        cv2.line(img, (0, top_line), (clo, top_line), (0, 255, 0))
        cv2.line(img, (0, bottom_line), (clo, bottom_line), (0, 255, 0))
        cv2.imwrite('test_line.jpg', img)
    return top_line, bottom_line


def clo_edge_detect(img_path):
    left_line = 0
    right_line = 0
    img = cv2.imread(img_path)
    row = img.shape[0]
    clo = img.shape[1]
    avg = mean(mean(img,axis=0),axis=1)
    left_line = nonzero(avg[:] < 62)[0][0]
    right_line = (intersect1d(nonzero(avg[:]>230)[0], nonzero(avg[:]<248)[0])).tolist()[-1]
    left_line += 3
    if False:# debug code
        img = cv2.imread('test.jpg')
        cv2.line(img, (left_line, 0), (left_line, row), (0, 255, 0))
        cv2.line(img, (right_line, 0), (right_line, row), (0, 255, 0))
        cv2.imwrite('test_line.jpg', img)
    return left_line, right_line


def valid_crop(img_path):
    name_lst = get_data_lst(img_path)
    for item in name_lst:
        top, bottom = row_edge_detect(img_path + item)
        left, right = clo_edge_detect(img_path + item)
        if top and bottom and left and right:return
        img = cv2.imread(img_path + item)
        crop_img = img[top:bottom, left:right]
        cv2.imwrite(img_path + item, crop_img)


if __name__ == '__main__':
    #trans2spec('../data/original_source/p_test.wav', '../data/validate/', 1)
    #valid_crop('../data/validate/')
    gen_data_txt_crop50('../data/train/crop50ms/')
    pass
