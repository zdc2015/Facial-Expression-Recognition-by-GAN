# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:42:47 2019

@author: zdc
"""

import numpy as np
import tensorflow as tf
import math


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# 构造可训练参数
def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)


# 定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output

# 定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding="SAME", name="atrous_conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        return output


# 定义lrelu激活层
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

# 使用tf.layers改写的generator网络结构，结果训练的时候出错了，是卷积层的问题，不知道为什么
def generator_1(image, reuse=False, isTrain=True, dropout_rate=0.5, num_classes=7):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0)

        # output [None, 32, 32, 64]
        conv1 = tf.layers.conv2d(image, 64, kernel_size=[4,4], strides=(2,2), padding='same',
                                 kernel_initializer=w_init, bias_constraint=b_init)
        bn1 = tf.layers.batch_normalization(conv1, training=isTrain)
        lrelu1 = lrelu(bn1, 0.2)

        # output [None, 16, 16, 128]
        conv2 = tf.layers.conv2d(lrelu1, 128, kernel_size=[4, 4], strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_constraint=b_init)
        bn2 = tf.layers.batch_normalization(conv2, training=isTrain)
        lrelu2 = lrelu(bn2, 0.2)

        # output [None, 8, 8, 256]
        conv3 = tf.layers.conv2d(lrelu2, 256, kernel_size=[4, 4], strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_constraint=b_init)
        bn3 = tf.layers.batch_normalization(conv3, training=isTrain)
        lrelu3 = lrelu(bn3, 0.2)

        # output [None, 4, 4, 512]
        conv4 = tf.layers.conv2d(lrelu3, 512, kernel_size=[4, 4], strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_constraint=b_init)
        bn4 = tf.layers.batch_normalization(conv4, training=isTrain)
        lrelu4 = lrelu(bn4, 0.2)

        # output [None, 2, 2, 512]
        conv5 = tf.layers.conv2d(lrelu4, 512, kernel_size=[4, 4], strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_constraint=b_init)
        bn5 = tf.layers.batch_normalization(conv5, training=isTrain)
        lrelu5 = lrelu(bn5, 0.2)

        # output [None, 1, 1, 512]
        conv6 = tf.layers.conv2d(lrelu5, 512, kernel_size=[4, 4], strides=(2, 2), padding='same',
                                 kernel_initializer=w_init, bias_constraint=b_init)
        bn6 = tf.layers.batch_normalization(conv6, training=isTrain)
        # lrelu6 = lrelu(bn6)

        # output [None, 2, 2, 512]
        decov1 = tf.layers.conv2d_transpose(tf.nn.relu(bn6), 512, [4,4], (2,2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov1 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov1,dropout_rate)), bn5],3)

        # output [None, 4, 4, 512]
        decov2 = tf.layers.conv2d_transpose(tf.nn.relu(decov1), 512, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov2 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov2, dropout_rate)), bn4], 3)

        # output [None, 8, 8, 256]
        decov3 = tf.layers.conv2d_transpose(tf.nn.relu(decov2), 256, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov3 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov3, dropout_rate)), bn3], 3)

        # output [None, 16, 16, 128]
        decov4 = tf.layers.conv2d_transpose(tf.nn.relu(decov3), 128, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov4 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov4, dropout_rate)), bn2], 3)

        # output [None, 32, 32, 64]
        decov5 = tf.layers.conv2d_transpose(tf.nn.relu(decov4), 64, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)

        # output [None, 64, 64, 3]
        decov6 = tf.layers.conv2d_transpose(tf.nn.relu(decov5), 3, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)

        print_activations(conv1)
        print_activations(conv2)
        print_activations(conv3)
        print_activations(conv4)
        print_activations(conv5)
        print_activations(conv6)
        print_activations(decov1)
        print_activations(decov2)
        print_activations(decov3)
        print_activations(decov4)
        print_activations(decov5)
        print_activations(decov6)

    with tf.variable_scope('CNN', reuse=reuse):
        flatten = tf.contrib.layers.flatten(bn6)
        fc = tf.contrib.layers.fully_connected(flatten, activation_fn=None, num_outputs=num_classes)

        residues = {'d1': decov1,
                    'd2': decov2,
                    'd3': decov3,
                    'd4': decov4,
                    'fc': fc,
                    }

        return tf.nn.tanh(decov6), residues


# 定义生成器，采用UNet架构，主要由8个卷积层和8个反卷积层组成
def generator(image, gf_dim=64, reuse=False, name="generator", num_classes=7, batch_size=10):
    input_dim = int(image.get_shape()[-1])  # 获取输入通道
    dropout_rate = 0.5  # 定义dropout的比例
    with tf.variable_scope(name,reuse=reuse):

        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0)

        # 第一个卷积层，输出尺度[None, 32, 32, 64]
        e1 = batch_norm(conv2d(input_=image, output_dim=gf_dim, kernel_size=4, stride=2, name='g_e1_conv'),
                        name='g_bn_e1')
        # 第二个卷积层，输出尺度[None, 16, 16, 128]
        e2 = batch_norm(conv2d(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_e2_conv'),
                        name='g_bn_e2')
        # 第三个卷积层，输出尺度[None, 8, 8, 256]
        e3 = batch_norm(conv2d(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_e3_conv'),
                        name='g_bn_e3')
        # 第四个卷积层，输出尺度[None, 4, 4, 512]
        e4 = batch_norm(conv2d(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e4_conv'),
                        name='g_bn_e4')
        # 第五个卷积层，输出尺度[None,2,2,512]
        e5 = batch_norm(conv2d(input_=lrelu(e4),output_dim=gf_dim*8,kernel_size=4,stride=2,name='g_e5_conv'),
                        name='g_bn_e5')
        # 第六个卷积层，输出尺度[None,1,1,512]
        e6 = batch_norm(conv2d(input_=lrelu(e5),output_dim=gf_dim*8,kernel_size=4,stride=2,name='g_e6_conv'),
                        name='g_bn_e6')

        # output [None, 2, 2, 512]
        decov1 = tf.layers.conv2d_transpose(tf.nn.relu(e6), 512, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov1 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov1, dropout_rate)), e5], 3)

        # output [None, 4, 4, 512]
        decov2 = tf.layers.conv2d_transpose(tf.nn.relu(decov1), 512, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov2 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov2, dropout_rate)), e4], 3)

        # output [None, 8, 8, 256]
        decov3 = tf.layers.conv2d_transpose(tf.nn.relu(decov2), 256, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov3 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov3, dropout_rate)), e3], 3)

        # output [None, 16, 16, 128]
        decov4 = tf.layers.conv2d_transpose(tf.nn.relu(decov3), 128, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)
        decov4 = tf.concat([tf.layers.batch_normalization(tf.layers.dropout(decov4, dropout_rate)), e2], 3)

        # output [None, 32, 32, 64]
        decov5 = tf.layers.conv2d_transpose(tf.nn.relu(decov4), 64, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)

        # output [None, 64, 64, 3]
        decov6 = tf.layers.conv2d_transpose(tf.nn.relu(decov5), input_dim, [4, 4], (2, 2), padding='same',
                                            kernel_initializer=w_init, bias_initializer=b_init)

    with tf.variable_scope('CNN', reuse=reuse):
        flatten = tf.contrib.layers.flatten(e6)
        fc = tf.contrib.layers.fully_connected(flatten, activation_fn=None, num_outputs=num_classes)

        residues = {'d1': decov1,
                    'd2': decov2,
                    'd3': decov3,
                    'd4': decov4,
                    'fc': fc,
                    }

        return tf.nn.tanh(decov6), residues
        # # 第一个反卷积层，输出尺度[None, 2, 2, 512]
        # d1 = deconv2d(input_=tf.nn.relu(e6), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d1', batch_size=batch_size)
        # d1 = tf.nn.dropout(d1, dropout_rate)  # 随机扔掉一般的输出
        # d1 = tf.concat([batch_norm(d1, name='g_bn_d1'), e5], 3)
        #
        # # 第二个反卷积层，输出尺度[None, 4, 4, 512]
        # d2 = deconv2d(input_=tf.nn.relu(d1), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d2', batch_size=batch_size)
        # d2 = tf.nn.dropout(d2, dropout_rate)  # 随机扔掉一般的输出
        # d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), e4], 3)
        #
        # # 第三个反卷积层，输出尺度[None, 8, 8, 256]
        # d3 = deconv2d(input_=tf.nn.relu(d2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_d3', batch_size=batch_size)
        # d3 = tf.nn.dropout(d3, dropout_rate)  # 随机扔掉一般的输出
        # d3 = tf.concat([batch_norm(d3, name='g_bn_d3'), e3], 3)
        #
        # # 第四个反卷积层，输出尺度[1, 16, 16, 128]
        # d4 = deconv2d(input_=tf.nn.relu(d3), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_d4', batch_size=batch_size)
        # d4 = tf.concat([batch_norm(d4, name='g_bn_d4'), e2], 3)
        #
        # # 第五个反卷积层，输出尺度[1, 32, 32, 64]
        # d5 = deconv2d(input_=tf.nn.relu(d4), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d5', batch_size=batch_size)
        #
        # # 第六个反卷积层，输出尺度[1, 64, 64, 3]
        # d6 = deconv2d(input_=tf.nn.relu(d5), output_dim=input_dim, kernel_size=4, stride=2, name='g_d6', batch_size=batch_size)
    #
    # with tf.variable_scope('CNN',reuse=reuse):
    #
    #     flatten = tf.contrib.layers.flatten(e6)
    #     fc = tf.contrib.layers.fully_connected(flatten, activation_fn=None, num_outputs=num_classes)
    #
    #     residues = {'d2': d2,
    #                 'd3': d3,
    #                 'd4': d4,
    #                 'd1': d1,
    #                 'fc': fc,
    #                 'e6': e6,
    #                 }
    #
    #     return tf.nn.tanh(d6), residues


# 定义判别器
def discriminator(image, targets, df_dim=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name,reuse=reuse):
        dis_input = tf.concat([image, targets], 3)
        # 第1个卷积模块，输出尺度: None*32*32*64
        h0 = lrelu(conv2d(input_=dis_input, output_dim=df_dim, kernel_size=4, stride=2, name='d_h0_conv'))

        # 第2个卷积模块，输出尺度: None*16*16*128
        h1 = lrelu(batch_norm(conv2d(input_=h0, output_dim=df_dim * 2, kernel_size=4, stride=2, name='d_h1_conv'),
                              name='d_bn1'))

        # 第3个卷积模块，输出尺度: None*8*8*256
        h2 = lrelu(batch_norm(conv2d(input_=h1, output_dim=df_dim * 4, kernel_size=4, stride=2, name='d_h2_conv'),
                              name='d_bn2'))

        # 第4个卷积模块，输出尺度: 1*4*4*512
        h3 = lrelu(batch_norm(conv2d(input_=h2, output_dim=df_dim * 8, kernel_size=4, stride=1, name='d_h3_conv'),
                              name='d_bn3'))
        # 第5个卷积模块，输出尺度: 1*4*4*1
        h4 = lrelu(batch_norm(conv2d(input_=h3, output_dim=1, kernel_size=4, stride=1, name='d_h4_conv'),
                              name='d_bn3'))
        # 最后一个卷积模块，输出尺度: None*1*1*1
        output = conv2d(input_=h3, output_dim=1, kernel_size=4, stride=1, name='d_h5_conv', padding='VALID')


        dis_out = tf.sigmoid(output)  # 在输出之前经过sigmoid层，因为需要进行log运算
        return dis_out


def local_cnn_1(residue, input_1=None, input_2=None, name='CNN1', num_classes=7):
    with tf.variable_scope(name,reuse=False):
        # [None,16,16,128*2]
        # residue = tf.concat([input_1,input_2],axis=3)
        # 第一个卷积层，输出尺度[None, 8, 8, 512]
        e1 = batch_norm(conv2d(input_=residue, output_dim=512, kernel_size=4, stride=2, name='c_e1_conv'),
                        name='c_bn_e1')
        # 第二个卷积层，输出尺度[None,4,4,512]
        e2 = batch_norm(conv2d(input_=lrelu(e1), output_dim=512, kernel_size=4, stride=2, name='c_e2_conv'),
                        name='c_bn_e2')
        # 第三个卷积层，输出尺度[None,2,2,512]
        e3 = batch_norm(conv2d(input_=lrelu(e2),output_dim=512,kernel_size=4,stride=2,name='c_e3_conv')
                        ,name='c_bn_e3')
        # 第四个卷积层，输出尺度[None,1,1,512]
        c4 = conv2d(input_=lrelu(e3),output_dim=512,kernel_size=4,stride=2,name='c_c4_conv')
        flatten = tf.contrib.layers.flatten(c4)
        fc = tf.contrib.layers.fully_connected(flatten,activation_fn=None,num_outputs=num_classes)

        return fc


def local_cnn_2(residue, input_1=None, input_2=None, name='CNN2', num_classes=7):
    with tf.variable_scope(name,reuse=False):
        # [None,8,8,256*2]
        # residue = tf.concat([input_1,input_2],axis=3)
        # 第一个卷积层，输出尺度[None, 8, 8, 512]
        e1 = batch_norm(conv2d(input_=residue, output_dim=512, kernel_size=4, stride=1, name='c_e1_conv'),
                        name='c_bn_e1')
        # 第二个卷积层，输出尺度[None,4,4,512]
        e2 = batch_norm(conv2d(input_=lrelu(e1), output_dim=512, kernel_size=4, stride=2, name='c_e2_conv'),
                        name='c_bn_e2')
        # 第三个卷积层，输出尺度[None,2,2,512]
        e3 = batch_norm(conv2d(input_=lrelu(e2),output_dim=512,kernel_size=4,stride=2,name='c_e3_conv')
                        ,name='c_bn_e3')
        # 第四个卷积层，输出尺度[None,1,1,512]
        c4 = conv2d(input_=lrelu(e3),output_dim=512,kernel_size=4,stride=2,name='c_c4_conv')
        flatten = tf.contrib.layers.flatten(c4)
        fc = tf.contrib.layers.fully_connected(flatten,activation_fn=None,num_outputs=num_classes)


        return fc


def local_cnn_3(residue, input_1=None, input_2=None, name='CNN3', num_classes=7):
    with tf.variable_scope(name,reuse=False):
        # [None,4,4,512*2]
        # residue = tf.concat([input_1,input_2],axis=3)
        # 第一个卷积层，输出尺度[None, 4, 4, 512]
        e1 = batch_norm(conv2d(input_=residue, output_dim=512, kernel_size=3, stride=1, name='c_e1_conv'),
                        name='c_bn_e1')
        # 第二个卷积层，输出尺度[None,4,4,512]
        e2 = batch_norm(conv2d(input_=lrelu(e1), output_dim=512, kernel_size=3, stride=1, name='c_e2_conv'),
                        name='c_bn_e2')
        # 第三个卷积层，输出尺度[None,2,2,512]
        e3 = batch_norm(conv2d(input_=lrelu(e2), output_dim=512, kernel_size=4, stride=2, name='c_e3_conv')
                        ,name='c_bn_e3')

        # 第四个卷积层，输出尺度[None,1,1,512]
        c4 = conv2d(input_=lrelu(e3),output_dim=512,kernel_size=4,stride=2,name='c_c4_conv')
        flatten = tf.contrib.layers.flatten(c4)
        fc = tf.contrib.layers.fully_connected(flatten,activation_fn=None,num_outputs=num_classes)

        return fc


def local_cnn_4(residue, input_1=None, input_2=None, name='CNN4', num_classes=7):
    with tf.variable_scope(name,reuse=False):
        # [None,2,2,512*2]
        # residue = tf.concat([input_1,input_2],axis=3)
        # 第一个卷积层，输出尺度[None, 2, 2, 512]
        e1 = batch_norm(conv2d(input_=residue, output_dim=512, kernel_size=3, stride=1, name='c_e1_conv'),
                        name='c_bn_e1')
        # 第二个卷积层，输出尺度[None,1,1,512]
        c2 = conv2d(input_=lrelu(e1), output_dim=512, kernel_size=3, stride=2, name='c_e2_conv')

        flatten = tf.contrib.layers.flatten(c2)
        fc = tf.contrib.layers.fully_connected(flatten,activation_fn=None,num_outputs=num_classes)

        return fc


def small_test():
    A = tf.Variable(tf.truncated_normal([2, 64, 64, 3], stddev=0.1))
    B = tf.Variable(tf.truncated_normal([2, 64, 64, 3], stddev=0.1))
    X = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='input_picture')
    np.random.seed(1)
    batch_size = 10
    image = np.random.rand(batch_size,64,64,3)

    C, residues = generator(X, batch_size=batch_size)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    e6 = residues['e6']
    e6.eval(feed_dict={X:image}, session=sess)

if __name__ == '__main__':
    small_test()