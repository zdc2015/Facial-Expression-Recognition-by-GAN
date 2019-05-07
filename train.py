# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:22:53 2019

@author: zdc
"""

from __future__ import print_function

import argparse
from util import *
from net import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--snapshot_dir", default='./snapshots', help="path of snapshots")  # 保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs")  # 训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=64, help="load image size")  # 网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")  # 随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam')  # 学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  # 训练的epoch数量
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')  # adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200,
                    help="times to summary.")  # 训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=100, help="times to write.")  # 训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=250, help="times to save.")  # 训练中每过多少step保存模型(可训练参数)
parser.add_argument("--lamda_l1_weight", type=float, default=0.0, help="L1 lamda")  # 训练中L1_Loss前的乘数
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda")  # 训练中GAN_Loss前的乘数

parser.add_argument("--lambda_1", type=float, default=0.7)
parser.add_argument("--lambda_2", type=float, default=0.5)
parser.add_argument("--lambda_3", type=float, default=0.3)
parser.add_argument("--lambda_4", type=float, default=0.2)
parser.add_argument("--lambda_5", type=float, default=1.0)



args = parser.parse_args()  # 用来解析命令行参数
EPS = 1e-12  # EPS用于保证log函数里面的参数大于零

minibatch_size = 256
train_classfication_epoch = 200
emotions = ['anger','contempt','disgust', 'fear', 'happy','neutral', 'sadness', 'surprise']

def save(saver, sess, logdir, step):  # 保存模型的save函数
    model_name = 'model'  # 保存的模型名前缀
    checkpoint_path = os.path.join(logdir, model_name)  # 模型的保存路径与名称
    if not os.path.exists(logdir):  # 如果路径不存在即创建
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)  # 保存模型
    print('The checkpoint has been created.')


def cv_inv_proc(img):  # cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32)  # 返回bgr格式的图像，方便cv2写图像


def get_write_picture(picture, gen_label, label, height, width):  # get_write_picture函数得到训练过程中的可视化结果
    picture_image = cv_inv_proc(picture)  # 还原输入的图像
    gen_label_image = cv_inv_proc(gen_label[0])  # 还原生成的样本
    label_image = cv_inv_proc(label)  # 还原真实的样本(标签)
    inv_picture_image = cv2.resize(picture_image, (width, height))  # 还原图像的尺寸
    inv_gen_label_image = cv2.resize(gen_label_image, (width, height))  # 还原生成的样本的尺寸
    inv_label_image = cv2.resize(label_image, (width, height))  # 还原真实的样本的尺寸
    output = np.concatenate((inv_picture_image, inv_gen_label_image, inv_label_image), axis=1)  # 把他们拼起来
    return output

def l1_loss(src, dst):  # 定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))

def smalltest():  # 训练程序的主函数
    if not os.path.exists(args.snapshot_dir):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir):  # 如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)

    picture, target, label, num_classes = get_ck_dataset()
    label = convert_to_one_hot(label, num_classes)
    picture = picture / 127.5 - 1.  # 归一化图片
    target = target / 127.5 - 1.  # 归一化图片

    num_train = minibatch_size*150
    num_test = minibatch_size*4

    train_picture = picture[:num_train]
    test_picture = picture[num_train:num_train+num_test]
    train_target = target[:num_train]
    test_target = target[num_train:num_train+num_test]
    train_label = label[:num_train]
    test_label = label[num_train:num_train+num_test]

    total_losses = []
    dis_losses = []
    gen_losses = []
    train_accs = []
    test_accs = []

    print('num of train data: %d' % (train_picture.shape[0]))
    print('num of test data: %d' % (test_label.shape[0]))
    print('input picture size:',train_picture.shape)

    input_picture = tf.placeholder(tf.float32, shape=[None, args.image_size, args.image_size, 3],
                                   name='train_picture')  # 输入的训练图像
    target_picture = tf.placeholder(tf.float32, shape=[None, args.image_size, args.image_size, 3],
                                 name='target_picture')  # 输入的与训练图像匹配的标签
    input_label = tf.placeholder(tf.float32, shape=[None, num_classes], name='train_label')

    gen_picture, residues = generator(image=input_picture, reuse=False, num_classes=num_classes)  # 得到生成器的输出
    dis_real = discriminator(image=input_picture, targets=target_picture, df_dim=64, reuse=False,
                             name="discriminator")  # 判别器返回的对真实标签的判别结果
    dis_fake = discriminator(image=input_picture, targets=gen_picture, df_dim=64, reuse=True,
                             name="discriminator")  # 判别器返回的对生成(虚假的)标签判别结果

    FC1 = local_cnn_1(residues['d4'], num_classes=num_classes)
    FC2 = local_cnn_2(residues['d3'], num_classes=num_classes)
    FC3 = local_cnn_3(residues['d2'], num_classes=num_classes)
    FC4 = local_cnn_4(residues['d1'], num_classes=num_classes)

    classification = tf.concat([FC1, FC2, FC3, FC4, residues['fc']],axis=1)
    with tf.variable_scope('CNN/classification', reuse=False):
        flatten = tf.contrib.layers.flatten(classification)
        classification = tf.contrib.layers.fully_connected(flatten, activation_fn=None, num_outputs=num_classes)


    # classification = FC1 + FC2 + FC3 + FC4 + residues['fc']


    # 计算准确度
    # predict_op = tf.arg_max(classification, 1)
    # correct_prediction = tf.equal(predict_op, tf.argmax(input_label, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    predict_op = tf.argmax(classification, axis=1)
    correct_prediction = tf.equal(predict_op, tf.argmax(input_label, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC1, labels=input_label))
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC2, labels=input_label))
    loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC3, labels=input_label))
    loss_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC4, labels=input_label))
    loss_5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classification, labels=input_label))
    total_loss = args.lambda_1 * loss_1 + args.lambda_2 * loss_2 + args.lambda_3 * loss_3 + args.lambda_4 * loss_4 + args.lambda_5 * loss_5

    gen_loss_GAN = tf.reduce_mean(-tf.log(dis_fake + EPS))  # 计算生成器损失中的GAN_loss部分
    gen_loss_L1 = tf.reduce_mean(l1_loss(gen_picture, target_picture))  # 计算生成器损失中的L1_loss部分
    gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_L1 * args.lamda_l1_weight  # 计算生成器的loss

    dis_loss = tf.reduce_mean(-(tf.log(dis_real + EPS) + tf.log(1 - dis_fake + EPS)))  # 计算判别器的loss

    gen_loss_sum = tf.summary.scalar("gen_loss", gen_loss)  # 记录生成器loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)  # 记录判别器loss的日志

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())  # 日志记录器

    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]  # 所有生成器的可训练参数
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]  # 所有判别器的可训练参数
    c_vars = [v for v in tf.trainable_variables() if 'CNN' in v.name]  # 所有分类模型的可训练参数
    restore_var = g_vars.copy()       # 不能直接restore_var = g_vars.extend(d_vars)
    restore_var = restore_var.extend(d_vars)

    d_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)  # 判别器训练器
    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)  # 生成器训练器
    c_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)  # 分类器训练器
    # G_optim = tf.train.AdamOptimizer(args.base_lr, beta1=0.5).minimize(gen_loss, var_list=g_vars)

    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars)  # 计算判别器参数梯度
    d_train = d_optim.apply_gradients(d_grads_and_vars)  # 更新判别器参数
    g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars)  # 计算生成器参数梯度
    g_train = g_optim.apply_gradients(g_grads_and_vars)  # 更新生成器参数
    c_grads_and_vars = c_optim.compute_gradients(total_loss, var_list=c_vars) # 计算分类器参数梯度
    c_train = c_optim.apply_gradients(c_grads_and_vars)  # 更新分类器参数

    train_op = tf.group(d_train, g_train)  # train_op表示了参数更新操作
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    sess = tf.Session(config=config)  # 新建会话层
    init = tf.global_variables_initializer()  # 参数初始化器
    sess.run(init)  # 初始化所有可训练参数
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)  # 模型保存
    # saver_re = tf.train.Saver(var_list=restore_var, max_to_keep=3)  # generator模型保存

    counter = 0  # counter记录训练步数

    flag = 0 # flag 标记
    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        flag = 1

    # 训练generator
    for epoch in range(args.epoch):  # 训练epoch数
        if flag == 1:
            break
        minibatch_num = train_picture.shape[0]//minibatch_size
        minibatchs = random_minibatch(train_picture, train_target, minibatch_size)
        epoch_gen_loss = 0
        epoch_dis_loss = 0
        for minibatch in minibatchs:  # 每个训练epoch中的训练step数
            counter += 1
            (tx, ty) = minibatch
            feed_dict = {input_picture: tx, target_picture: ty}  # 构造feed_dict
            gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op],
                                                         feed_dict=feed_dict)  # 得到每个step中的生成器和判别器loss
            epoch_gen_loss += gen_loss_value/minibatch_num
            epoch_dis_loss += dis_loss_value/minibatch_num

            if counter % args.save_pred_every == 0:  # 每过save_pred_every次保存模型
                save(saver, sess, args.snapshot_dir, counter)
            if counter % args.summary_pred_every == 0:  # 每过summary_pred_every次保存训练日志
                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, dis_loss_sum],
                                                                       feed_dict=feed_dict)
                summary_writer.add_summary(gen_loss_sum_value, counter)
                summary_writer.add_summary(discriminator_sum_value, counter)
            if counter % args.write_pred_every == 0:  # 每过write_pred_every次写一下训练的可视化结果
                gen_label_value = sess.run(gen_picture, feed_dict=feed_dict)  # run出生成器的输出
                write_image = get_write_picture(tx[0], gen_label_value, ty[0], args.image_size,
                                                args.image_size)  # 得到训练的可视化结果
                write_image_name = args.out_dir + "/out" + str(counter) + ".png"  # 待保存的训练可视化结果路径与名称
                cv2.imwrite(write_image_name, write_image)  # 保存训练的可视化结果
            print('epoch {:d} step {:d} \t gen_loss = {:.5f}, dis_loss = {:.5f}'.format(epoch, counter, gen_loss_value,
                                                                                        dis_loss_value))
        gen_losses.append(epoch_gen_loss)
        dis_losses.append(epoch_dis_loss)
        print_gen_dis_loss(gen_losses=gen_losses, dis_losses=dis_losses)

    for epoch in range(train_classfication_epoch):
        epoch_cost = 0
        minibatch_num = train_label.shape[0] // minibatch_size
        minibatchs = random_minibatch(train_picture, train_label, minibatch_size)
        print('epoch {}'.format(epoch))
        for minibatch in tqdm(minibatchs):
            counter += 1
            (tx, ty) = minibatch
            feed_dict = {input_picture: tx, input_label: ty}  # 构造feed_dict
            loss, _ = sess.run([total_loss, c_train], feed_dict=feed_dict)  # 训练total loss
            epoch_cost+=loss/minibatch_num

        test_sets = random_minibatch(test_picture, test_label, minibatch_size)
        num_test_sets = len(test_sets)
        acc = 0
        for test_set in test_sets:
            (tx, ty) = test_set
            t = accuracy.eval(session=sess, feed_dict={input_picture:tx, input_label:ty})
            acc += t/num_test_sets

        print("Cost, test_accuracy after epoch %i: %f, %f" % (epoch, epoch_cost, acc))
        test_accs.append(acc)
        total_losses.append(epoch_cost)
        print_hist(total_losses, ylabel='total_loss', name='total_loss')
        print_hist(test_accs, ylabel='test_acc', name='test_acc')
        if epoch % 10 == 0:
            save(saver, sess, args.snapshot_dir, counter)

if __name__ == '__main__':
    smalltest()