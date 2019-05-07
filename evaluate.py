
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

minibatch_size = 1
emotions = ['anger','contempt','disgust', 'fear', 'happy', 'sadness', 'surprise']
eval_out_dir = r'.\eval_out'

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

def test_ck():
    picture, target, label, num_classes = get_ck_dataset()
    label = convert_to_one_hot(label, num_classes)
    picture = picture / 127.5 - 1.  # 归一化图片
    target = target / 127.5 - 1.  # 归一化图片

    num_train = minibatch_size * 256 * 150
    num_test = 1024 * 10

    train_picture = picture[:num_train]
    test_picture = picture[num_train:num_train + num_test]
    train_target = target[:num_train]
    test_target = target[num_train:num_train + num_test]
    train_label = label[:num_train]
    test_label = label[num_train:num_train + num_test]

    picture = test_picture
    target = test_target
    label = test_label
    return picture, target, label, num_classes

def eval():
    if not os.path.exists(eval_out_dir):  # 如果保存测试中可视化输出的文件夹不存在则创建
        os.makedirs(eval_out_dir)

    # picture, target, label, num_classes = get_TFEID_dataset()
    # label = convert_to_one_hot(label, num_classes)
    # picture = picture / 127.5 - 1.  # 归一化图片
    # target = target / 127.5 - 1.  # 归一化图片

    picture, target, label, num_classes = test_ck()

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

    classification = tf.concat([FC1, FC2, FC3, FC4, residues['fc']], axis=1)
    with tf.variable_scope('CNN/classification', reuse=False):
        flatten = tf.contrib.layers.flatten(classification)
        classification = tf.contrib.layers.fully_connected(flatten, activation_fn=None, num_outputs=num_classes)

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    sess = tf.Session(config=config)  # 新建会话层
    init = tf.global_variables_initializer()  # 参数初始化器
    sess.run(init)  # 初始化所有可训练参数

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)  # 模型保存
    checkpoint = tf.train.latest_checkpoint(args.snapshot_dir)  # 读取模型参数
    saver.restore(sess, checkpoint)  # 导入模型参数

    counter = 0
    total_acc = 0
    minibatch_num = picture.shape[0] / minibatch_size
    minibatchs = random_minibatch3(picture, label, target, minibatch_size)
    for minibatch in tqdm(minibatchs):
        counter +=1
        (tx, ty, tz) = minibatch
        gen, predict, acc = sess.run([gen_picture, predict_op, accuracy],feed_dict={input_picture: tx, input_label: ty})
        total_acc += acc / minibatch_num
        write_image = get_write_picture(tx[0], gen, tz[0], args.image_size,
                                        args.image_size)  # 得到训练的可视化结果
        real_emotions = np.argmax(ty, axis=1)
        # write_image_name = str(counter)+'.png'
        if predict == real_emotions[0]:
            write_image_name = eval_out_dir + "\\{}predict_{}".format(counter, emotions[predict[0]]) + "__real_{}".format(emotions[int(real_emotions[0])])+".png"  # 待保存的训练可视化结果路径与名称
        else:
            write_image_name = eval_out_dir + "\!!!!!!!!!!" + "{}predict_{}".format(counter, emotions[predict[0]]) + "__real_{}".format(emotions[int(real_emotions[0])]) + ".png"
        flag = cv2.imwrite(write_image_name, write_image)  # 保存训练的可视化结果. 卧槽为什么会失败，尼玛的,好吧，命名错误
        if flag == False:
            break
    print("eval accuracy:{}".format(total_acc))
    sess.close()

if __name__ =="__main__":
    eval()