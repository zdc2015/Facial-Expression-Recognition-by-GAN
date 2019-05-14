import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from glob import glob
from tqdm import tqdm
import cv2

cascade = cv2.CascadeClassifier(r".\haarcascade_frontalface_alt2.xml")

expression_index = {'anger':0,'contempt':1,'disgust':2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6}

# 检测人脸，返回人脸坐标
def detect(img,cascade):
    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gimg, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        print('error: no faces')
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def get_head_image(img, reshape_size = 64):
    rects = detect(img, cascade)
    x1,y1,x2,y2 = rects[0]
    cropped = img[y1:y2,x1:x2]
    cropped_re = cv2.resize(cropped, (70,70))[3:67, 3:67]
    cv2.imwrite('1.jpg', cropped_re)
    return cropped_re

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def convert_to_one_hot(Y, classes):
    one_hot = np.eye(classes)[Y.reshape(-1)]
    return one_hot

def print_bar(x, y, name_list, xlabel='expression', ylabel='probability'):
    plt.bar(range(x), height=y, tick_label=name_list, color='rgb')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    name = '1.png'
    plt.savefig(name)
    plt.close()
    return cv2.imread(name)

def print_hist(losses, xlabel='epochs', ylabel='loss', name='loss'):
    plt.plot(losses,color='blue', linestyle='-', linewidth=1.0, label='gen_loss')
    plt.title(ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xlim(0, num_epoch)
    # plt.axis([0,200,-5,5]) # [xmin xmax ymin ymax]

    plt.savefig(name+'.png')
    plt.close()

def print_gen_dis_loss(gen_losses, dis_losses, name='gen and dis loss'):
    l1, = plt.plot(gen_losses, color='blue', linestyle='-', linewidth=1.0, label='gen_loss')
    l2, = plt.plot(dis_losses, color='red', linestyle='-', linewidth=1.0, label='dis_loss')
    plt.legend(handles=[l1, l2, ], loc='best')

    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.xlim(0, num_epoch)
    plt.savefig(name+'.png')
    plt.close()

def compute_cost(Result, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Result,labels=Y))
    return cost

def get_target_picture(file_name, targets_path, databases='ck+'):
    if databases=='ck+':
        targets = glob(os.path.join(targets_path,file_name[:-7]+"*.png"))
        if len(targets)==0:
            print(os.path.join(targets_path, file_name[:-7],"*.png"))
        else:
            return targets[0]
    elif databases=='TFEID High':
        targets = glob(os.path.join(targets_path, file_name[:4]+"*.jpg"))
        if len(targets)==0:
            print(os.path.join(targets_path, file_name[:4]+"*.jpg"))
        else:
            return targets[0]
    elif databases=='jaffe':
        targets = glob(os.path.join(targets_path, file_name[:3] + "*"))
        if len(targets) == 0:
            print(os.path.join(targets_path, file_name[:3] + "*"))
        else:
            return targets[0]

def pre_process_onepicture(img):
    img = cv2.imread(img)
    img_crop = get_head_image(img)
    img_crop_ex = np.expand_dims(img_crop, axis=0)
    return img_crop_ex

# def get_dataset(dataset_name='CK+ rearranged rotated cropped5', img_size = 64, seed=1234):
#     #datasets_path = r'C:\Users\zdc\Downloads\FER\dataset'
#     # base_path = r'F:\space\dataset'
#     datasets_path = '.\\dataset\\'
#
#     if dataset_name == 'CK+ rearranged rotated cropped5':
#         dataset_path = os.path.join(datasets_path, dataset_name)
#         target_picture_path = os.path.join(dataset_path, 'neutral')
#         dirs = os.listdir(dataset_path)
#         i = 0
#         j = 0
#
#         train_picture = None
#         target_picture = None
#         train_label = None
#
#         # 选择80%的图片作为训练集
#         # 获取训练集
#         for dir in dirs:
#             filespath = os.path.join(dataset_path, dir, "*.png")
#             pics = glob.glob(filespath)
#             for pic in tqdm.tqdm(pics):
#                 if j == 0:
#                     train_picture = cv2.imread(pic)
#                     # train_picture = cv2.resize(train_picture, (size, size))
#                     train_picture = np.expand_dims(train_picture, axis=0)
#
#                     target_pic = get_target_picture(os.path.basename(pic), target_picture_path)
#                     target_picture = cv2.imread(target_pic)
#                     # target_picture = cv2.resize(target_picture, (size, size))
#                     target_picture = np.expand_dims(target_picture, axis=0)
#
#                     train_label = np.array([[i]])
#                 elif j < 10:
#                     t = cv2.imread(pic)
#                     # t = cv2.resize(t, (size, size))
#                     t = np.expand_dims(t, axis=0)
#
#                     target_pic = get_target_picture(os.path.basename(pic), target_picture_path)
#                     target_t = cv2.imread(target_pic)
#                     # target_t = cv2.resize(target_t, (size, size))
#                     target_t = np.expand_dims(target_t, axis=0)
#
#                     train_picture = np.concatenate((train_picture, t), axis=0)
#                     target_picture = np.concatenate((target_picture, target_t), axis=0)
#                     train_label = np.concatenate((train_label, np.array([[i]])), axis=0)
#                 j += 1
#             i += 1
#
#
#         np.random.seed(seed)
#         random_index = list(np.random.permutation(train_picture.shape[0]))
#         train_label = train_label[random_index]
#         train_picture = train_picture[random_index]
#         target_picture = target_picture[random_index]
#
#         return train_picture, target_picture, train_label, i

def get_ck_dataset(seed = 1234):
    #datasets_path = r'C:\Users\zdc\Downloads\FER\dataset'
    datasets_path = r'.\dataset'
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    j = 0
    i = 0
    train_picture = None
    target_picture = None
    train_label = None
    for emotion in emotions:
        if j == 0:
            train_picture = np.load(os.path.join(datasets_path,emotion+'_input.npy'))
            #train_picture = train_picture.reshape((-1,64,64,3))
            #train_picture = np.expand_dims(train_picture, axis=0)

            target_picture = np.load(os.path.join(datasets_path,emotion+'_target.npy'))
            # target_picture = cv2.resize(target_picture, (size, size))
            #target_picture = np.expand_dims(target_picture, axis=0)

            train_label = np.load(os.path.join(datasets_path,emotion+'_label.npy'))
        else:
            t = np.load(os.path.join(datasets_path,emotion+'_input.npy'))
            #t = t.reshape((-1, 64, 64, 3))
            #t = np.expand_dims(t, axis=0)

            target_t = np.load(os.path.join(datasets_path,emotion+'_target.npy'))
            #target_t = np.expand_dims(target_t, axis=0)

            label_t = np.load(os.path.join(datasets_path,emotion+'_label.npy'))
            # print(train_picture.shape)
            # print(t.shape)
            train_picture = np.concatenate((train_picture, t), axis=0)
            target_picture = np.concatenate((target_picture, target_t), axis=0)
            train_label = np.concatenate((train_label, label_t), axis=0)
        j += 1
    np.random.seed(seed)
    random_index = list(np.random.permutation(train_picture.shape[0]))
    train_label = train_label[random_index]
    train_picture = train_picture[random_index]
    target_picture = target_picture[random_index]

    return train_picture, target_picture, train_label, j

def get_ck_rearrange_crop(seed=1234, size=64):
    dataset_path = r'..\dataset\CK+ rearranged cropped'
    target_img_dir_path = dataset_path + r'\neutral'
    dirs_path = glob(dataset_path + '\*')
    train_picture = None
    target_picture = None
    train_label = None
    j = 0
    i = 0
    for dir_path in dirs_path:
        if dir_path == target_img_dir_path:
            continue
        imgs_path = glob(dir_path + '\*')
        for img_path in tqdm(imgs_path):
            if j == 0:
                train_picture = cv2.imread(img_path)
                train_picture = cv2.resize(train_picture, (size, size))
                train_picture = np.expand_dims(train_picture, axis=0)

                target_pic = get_target_picture(os.path.basename(img_path), target_img_dir_path, databases='ck+')
                target_picture = cv2.imread(target_pic)
                target_picture = cv2.resize(target_picture, (size, size))
                target_picture = np.expand_dims(target_picture, axis=0)

                train_label = np.array([[i]])
            else:
                t = cv2.imread(img_path)
                t = cv2.resize(t, (size, size))
                t = np.expand_dims(t, axis=0)

                target_pic = get_target_picture(os.path.basename(img_path), target_img_dir_path, databases='ck+')
                target_t = cv2.imread(target_pic)
                target_t = cv2.resize(target_t, (size, size))
                target_t = np.expand_dims(target_t, axis=0)

                train_picture = np.concatenate((train_picture, t), axis=0)
                target_picture = np.concatenate((target_picture, target_t), axis=0)
                train_label = np.concatenate((train_label, np.array([[i]])), axis=0)
            j += 1
        i += 1

    np.random.seed(seed)
    random_index = list(np.random.permutation(train_picture.shape[0]))
    train_label = train_label[random_index]
    train_picture = train_picture[random_index]
    target_picture = target_picture[random_index]

    return train_picture, target_picture, train_label, i

def get_TFEID_dataset(seed = 1234, size=64):
    dataset_path = r'..\dataset\TFEID High'
    target_img_dir_path = dataset_path + '\dfh_neutral_x'
    dirs_path = glob(dataset_path+'\*')
    train_picture = None
    target_picture = None
    train_label = None
    j = 0
    i = 0
    for dir_path in dirs_path:
        if dir_path==target_img_dir_path:
            continue
        imgs_path = glob(dir_path+'\*')
        for img_path in imgs_path:
            if j == 0:
                train_picture = cv2.imread(img_path)
                train_picture = cv2.resize(train_picture, (size, size))
                train_picture = np.expand_dims(train_picture, axis=0)

                target_pic = get_target_picture(os.path.basename(img_path), target_img_dir_path, databases='TFEID High')
                target_picture = cv2.imread(target_pic)
                target_picture = cv2.resize(target_picture, (size, size))
                target_picture = np.expand_dims(target_picture, axis=0)

                train_label = np.array([[i]])
            else:
                t = cv2.imread(img_path)
                t = cv2.resize(t, (size, size))
                t = np.expand_dims(t, axis=0)

                target_pic = get_target_picture(os.path.basename(img_path), target_img_dir_path, databases='TFEID High')
                target_t = cv2.imread(target_pic)
                target_t = cv2.resize(target_t, (size, size))
                target_t = np.expand_dims(target_t, axis=0)

                train_picture = np.concatenate((train_picture, t), axis=0)
                target_picture = np.concatenate((target_picture, target_t), axis=0)
                train_label = np.concatenate((train_label, np.array([[i]])), axis=0)
            j += 1
        i += 1

    np.random.seed(seed)
    random_index = list(np.random.permutation(train_picture.shape[0]))
    train_label = train_label[random_index]
    train_picture = train_picture[random_index]
    target_picture = target_picture[random_index]

    return train_picture, target_picture, train_label, i

def get_jaffe_dataset(seed = 1234, size=64):
    dataset_path = r'..\dataset\jaffe rearranged'
    target_img_dir_path = dataset_path + '\\neutral'
    dirs_path = glob(dataset_path + '\*')
    train_picture = None
    target_picture = None
    train_label = None
    j = 0
    i = 0
    for dir_path in dirs_path:
        if dir_path == target_img_dir_path:
            continue
        imgs_path = glob(dir_path + '\*')
        for img_path in imgs_path:
            if j == 0:
                train_picture = cv2.imread(img_path)
                train_picture = get_head_image(train_picture)
                train_picture = cv2.resize(train_picture, (size, size))
                train_picture = np.expand_dims(train_picture, axis=0)

                target_pic = get_target_picture(os.path.basename(img_path), target_img_dir_path, databases='jaffe')
                target_picture = cv2.imread(target_pic)
                target_picture = get_head_image(target_picture)
                target_picture = cv2.resize(target_picture, (size, size))
                target_picture = np.expand_dims(target_picture, axis=0)

                train_label = np.array([[expression_index[os.path.basename(dir_path)]]])
            else:
                t = cv2.imread(img_path)
                t = cv2.resize(t, (size, size))
                t = np.expand_dims(t, axis=0)

                target_pic = get_target_picture(os.path.basename(img_path), target_img_dir_path, databases='jaffe')
                target_t = cv2.imread(target_pic)
                target_t = cv2.resize(target_t, (size, size))
                target_t = np.expand_dims(target_t, axis=0)

                train_picture = np.concatenate((train_picture, t), axis=0)
                target_picture = np.concatenate((target_picture, target_t), axis=0)
                train_label = np.concatenate((train_label, np.array([[expression_index[os.path.basename(dir_path)]]])), axis=0)
            j += 1
        i += 1

    np.random.seed(seed)
    random_index = list(np.random.permutation(train_picture.shape[0]))
    train_label = train_label[random_index]
    train_picture = train_picture[random_index]
    target_picture = target_picture[random_index]

    return train_picture, target_picture, train_label, 7

def random_minibatch(X, Y, minibatch_size=64):
    m = X.shape[0]
    random_index = list(np.random.permutation(m))

    shuffle_X = X[random_index]
    shuffle_Y = Y[random_index]
    minibatchs = []

    num = math.floor(m/minibatch_size)
    for i in range(num):
        tx = shuffle_X[i*minibatch_size:(i+1)*minibatch_size]
        ty = shuffle_Y[i*minibatch_size:(i+1)*minibatch_size]
        minibatch = (tx, ty)
        minibatchs.append(minibatch)

    # if m%minibatch_size != 0:
    #     tx = shuffle_X[num*minibatch_size:]
    #     ty = shuffle_Y[num*minibatch_size:]
    #     minibatch = (tx, ty)
    #     minibatchs.append(minibatch)

    return minibatchs

def random_minibatch3(X, Y, Z, minibatch_size=64):
    m = X.shape[0]
    random_index = list(np.random.permutation(m))

    shuffle_X = X[random_index]
    shuffle_Y = Y[random_index]
    shuffle_Z = Z[random_index]
    minibatchs = []

    num = math.floor(m/minibatch_size)
    for i in range(num):
        tx = shuffle_X[i*minibatch_size:(i+1)*minibatch_size]
        ty = shuffle_Y[i*minibatch_size:(i+1)*minibatch_size]
        tz = shuffle_Z[i * minibatch_size:(i + 1) * minibatch_size]
        minibatch = (tx, ty, tz)
        minibatchs.append(minibatch)

    # if m%minibatch_size != 0:
    #     tx = shuffle_X[num*minibatch_size:]
    #     ty = shuffle_Y[num*minibatch_size:]
    #     minibatch = (tx, ty)
    #     minibatchs.append(minibatch)

    return minibatchs

def build_dataset(dataset_name='CK+ rearranged rotated cropped5', img_size = 64, seed=1234):
    #datasets_path = r'C:\Users\zdc\Downloads\FER\dataset'
    datasets_path = r'..\dataset'

    if dataset_name == 'CK+ rearranged rotated cropped5':
        dataset_path = os.path.join(datasets_path, dataset_name)
        target_picture_path = os.path.join(dataset_path, 'neutral')
        dirs = os.listdir(dataset_path)
        print(dirs)
        i = 0

        for dir in dirs:
            if dir == 'neutral':
                continue
            filespath = os.path.join(dataset_path, dir, "*.png")
            pics = glob(filespath)
            train_picture = None
            target_picture = None
            train_label = None
            j = 0
            for pic in tqdm(pics):
                if j == 0:
                    train_picture = cv2.imread(pic)
                    # train_picture = cv2.resize(train_picture, (size, size))
                    train_picture = np.expand_dims(train_picture, axis=0)

                    target_pic = get_target_picture(os.path.basename(pic), target_picture_path)
                    target_picture = cv2.imread(target_pic)
                    # target_picture = cv2.resize(target_picture, (size, size))
                    target_picture = np.expand_dims(target_picture, axis=0)

                    train_label = np.array([[i]])
                else:
                    t = cv2.imread(pic)
                    # t = cv2.resize(t, (size, size))
                    t = np.expand_dims(t, axis=0)

                    target_pic = get_target_picture(os.path.basename(pic), target_picture_path)
                    target_t = cv2.imread(target_pic)
                    # target_t = cv2.resize(target_t, (size, size))
                    target_t = np.expand_dims(target_t, axis=0)

                    train_picture = np.concatenate((train_picture, t), axis=0)
                    target_picture = np.concatenate((target_picture, target_t), axis=0)
                    train_label = np.concatenate((train_label, np.array([[i]])), axis=0)
                j += 1
            i += 1

            np.save(dir+'_input.npy',train_picture)
            np.save(dir + '_target.npy', target_picture)
            np.save(dir + '_label.npy', train_label)

if __name__ == '__main__':
    get_TFEID_dataset()