# Transformer
Substation transformer VPR

油浸式变压器负载&声纹关系

简介

在变电站工作一段时间的巡维人员能够通过声音（变压器噪声）分辨主变的负载高低。是否可以通过声纹识别相关
技术实现类似“耳朵”功能？


结构说明

data：   数据

docs：   设计、说明参考等文档

models： 模型存放目录

pub：    共用文件

src：    源码

V1.2 Instruction:

    1. 7 tfrecords data, image size: 600*800
    2. random crop image into 100*100 (my GPU:940MX, GPU Mermory: 1GB)
    3. tflearn have been used in this version
    4. validation accuarcy is 0.5 or so