import cv2
from numpy import *
import sys
sys.path.append('../pub/')
from file import *


def get_img_name():
    lst = []
    fw = open('../data/train/data.txt', 'r')
    for line in fw.readlines():
        line = line.split()
        lst.append(line[0])
    return lst


def crop_50ms():
    img_lst = get_data_lst('../data/train/spec/')
    #print(img_lst)
    for _name in img_lst:
        img = cv2.imread('../data/train/spec/'+_name)
        height = img.shape[0]
        #print(img.shape) # fixed width:614
        for n in range(10):
            crop_start = random.randint(20,400)
            crop_end = crop_start + 50
            if height<400:
                print('aaa')
            crop_img = img[height-400:height, crop_start:crop_end]
            cv2.imwrite('../data/train/crop50ms/'+_name[:-4]+ str(n)+ '.jpg', crop_img)

def check_size():
    img_lst = get_data_lst('../data/validate/')
    #print(img_lst)
    for _name in img_lst:
        img = cv2.imread('../data/validate/' + _name)
        if img is None:
            print(_name)
            continue
        height = img.shape[0]
        width = img.shape[1]
        if height < 400 or width < 400:
            print('Error_Size', _name)






if __name__ == '__main__':
    crop_50ms()
