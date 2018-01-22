from pylab import *
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy import *
import cv2
import os
import sys
sys.path.append('../../pub/')
from file import *


# transform wav file into specgram image
# V1.0
# def trans2spec(wav, store_path, label):
#     '''
#     :param wav: 源音频信号
#     :param store_path
#     :param label
#     :return: label
#     1.首先依据声强的差值关系简单确立起止点
#     2.再将每个crop转换成对应的语谱图
#     注意点：差值是根据经验确定的，不同的声纹是否能通用？
#     '''
#     # get voice sample frequency and data
#     sam_freq, snd = wavfile.read(wav)
#     #print(sam_freq)
#     sam_num, nchannels = shape(snd)
#     # use one channel
#     s1 = snd[:, 0]
#     len_wav = len(s1)
#     slot = 500  # crop wav into 500ms pieces
#     pace = int(float(slot) * sam_freq / 1000.0)
#
#     cnt_lst = []
#     for index in arange(0, len_wav, pace):
#         chuck = s1[index:index+pace]
#         if sum(abs(chuck)) < 1000000:
#             continue
#         cnt_lst.append(int(float(index) / pace))
#         plt.specgram(chuck, Fs=sam_freq, scale_by_freq=True, sides='default')
#         # plt.savefig(store_path + str(int(float(index)/pace)) + '_' + str(label) + '.jpg')
#         plt.axis('off')
#         plt.savefig('./123.png')
#         break
#     return cnt_lst


# V1.0
def trans2spec(wav, store_path):
    '''
    :param wav: 源音频信号
    :param store_path
    :return: label
    1.首先依据声强的差值关系简单确立起止点
    2.再将每个crop转换成对应的语谱图
    注意点：差值是根据经验确定的，不同的声纹是否能通用？
    '''
    name = wav.split('/')[-1][:-4]
    # get voice sample frequency and data
    sam_freq, snd = wavfile.read(wav)
    s1 = snd[:, 0]
    # print(len(s1))
    len_wav = len(s1)
    slot = 500  # crop wav into 500ms pieces
    pace = int(float(slot) * sam_freq / 1000.0)
    # print(sam_freq)

    cnt_lst = []
    # print(len_wav/pace)

    reg = 0
    for index in arange(0, len_wav, pace):
        chuck = s1[index:index+pace]
        if sum(abs(chuck)) < 1000000:
            continue
        plt.specgram(chuck, Fs=sam_freq, scale_by_freq=True, sides='default', cmap='gray',)
        plt.axis('off')
        plt.savefig(store_path+name+'_'+str(reg)+'.png')
        plt.cla()
        reg += 1
    return
        # break



if __name__ == '__main__':
    src_path = 'F:/CloudMusic/p/wav/'
    store_path = 'F:DL/data/vpr_data/'
    names = get_file_name(src_path)
    for name in names[:-10]:
        print(name)
        trans2spec(src_path+name, store_path+'train/')

    pass
