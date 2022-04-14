import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import cv2
import json

# def store_as_json():

def L_norm(x, lp=2):
    return np.linalg.norm(x, ord=lp, axis=-1)

def cal_distance(x1, x2, lp=2):
    dis = x1.reshape(x1.shape[0], -1) - x2.reshape(x2.shape[0], -1)
    dis = L_norm(dis, lp=lp)
    return dis

def softmax(x):
    x=np.exp(x)
    x_sum=np.sum(x, axis=1, keepdims=True)+0.0001
    return x/x_sum

def project(x, epsilon, lp):
    if lp>0 and lp !=np.inf:
        x_flat=tf.reshape(x, (x.shape[0],-1))
        x_norms=tf.norm(x_flat, ord=lp, axis=-1)
        x_norms=tf.maximum(x_norms, tf.ones_like(x_norms)*1e-12)
        factors=epsilon/x_norms
        factors=tf.minimum(factors, tf.ones_like(factors))
        x=tf.transpose(tf.transpose(x)*factors)
    elif lp==np.inf:
        x=tf.clip_by_value(x, -epsilon, epsilon)
    return x

def random_uniform(x, epsilon, lp):
    if lp>0 and lp !=np.inf:
        random_sate=np.random.uniform(-1,1,x.shape).astype('float32')
        random_sate=project(random_sate, epsilon, lp)
    elif lp==np.inf:
        random_sate=np.random.uniform(-epsilon, epsilon, x.shape).astype('float32')
    return x+random_sate

def normal_gradient(x, lp):
    if lp>0 and lp !=np.inf:
        x_flat = tf.reshape(x, (x.shape[0],-1))
        x_norms = tf.norm(x_flat, ord=lp, axis=-1)
        x_norms = tf.maximum(x_norms, tf.ones_like(x_norms)*1e-12)
        factors=1/x_norms
        x=tf.transpose(tf.transpose(x)*factors)
    elif lp==np.inf:
        x=tf.sign(x)
    return x

def getTimeStamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())


def buildDirs(address):
    isExists = os.path.exists(address)
    if not isExists:
        os.makedirs(address)

#self.saveImgPath+'/'+self.dataset.name+'_'+img_name+".png"
def plot_imgs(
        imgs,
        img_path='',
        r=5, c=5,
        img_show_flag=False,
        img_save_flag=True,
        figsize=None,
        randomFlag=True
):
#    imgs = 0.5 * imgs + 0.5
    if randomFlag:
        idx = np.random.randint(0, imgs.shape[0], r*c)
        imgs=imgs[idx]
    if figsize != None:
        plt.figure(figsize=figsize)
    # fig, axs = plt.subplots(r, c)
    # cnt = 0
    for i in range(r):
        for j in range(c):
            plt.subplot(r,c,i*c+j+1)
            plt.xticks([], [])
            plt.yticks([], [])
            if imgs.shape[3]==3:
                # axs[i, j].imshow(imgs[cnt, :, :, :])
                plt.imshow(imgs[i*c+j, :, :, :])
            else:
                # axs[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
                plt.imshow(imgs[i*c+j, :, :, 0], cmap='gray')
            # axs[i, j].axis('off')
            # cnt += 1
    plt.tight_layout()
    if img_save_flag:
        # fig.savefig(img_path)
        plt.savefig(img_path)
    if img_show_flag:
        plt.show()
    plt.clf()
    plt.close()

def readCsv(filePath, fileStart=None, fileEnd=None):
    if fileStart is not None:
        data=np.array(pd.read_csv(filePath,usecols=range(fileStart,fileEnd)))
    else:
        data=np.array(pd.read_csv(filePath))
    return data

def toCsv(data, fileName):
    filePara = pd.DataFrame(data=data)
    filePara.to_csv(fileName)

def listdir(path, fileType, flagRe=False):
    list_file=[]
    ld=os.listdir(path)
    for file in ld:
        file_path = os.path.join(path, file)
        if flagRe==True and os.path.isdir(file_path):
            list_file += listdir(file_path, fileType, flagRe=flagRe)
        if os.path.splitext(file)[1]==fileType:
            list_file.append(
                {
                    'name':os.path.splitext(file)[0],
                    'type':fileType,
                    'path':file_path
                }
            )
    return list_file

def affine_crd(img):
    rows, cols = img.shape[:2]
    crd=np.zeros((3,2))
    crd[0]=np.random.randint(0,rows/4,(1,2))
    crd[1]=np.random.randint(0,rows/4,(1,2))+np.array([rows/4*3,0])
    crd[2]=np.random.randint(0,rows/4,(1,2))+np.array([rows/2,rows/4*3])
    return crd.astype(np.float32)

def affine_trans(
        img,
        aff_flag=True,
        rota_flag=True
):
    rows, cols = img.shape[:2]
    res=img
    if aff_flag:
        pts1=affine_crd(img)
        pts2=affine_crd(img)
        M = cv2.getAffineTransform(pts1, pts2)
        res = cv2.warpAffine(img, M, (rows, cols))
    if rota_flag:
        degree = np.random.randint(0, 360, 1)[0]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
        res = cv2.warpAffine(res, M, (rows, cols))

    return res

def rotation_trans(img):
    rows, cols = img.shape[:2]
    degree = np.random.randint(0, 360, 1)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    res = cv2.warpAffine(img, M, (rows, cols))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(res)
    plt.show()

def add_noise(
            sample,
            eps=1,
            mean=0,
            std=1,
            l_flag=False,
            noise_only=False
    ):
        shape = sample.shape
        noise = np.random.normal(
            mean,
            std,
            shape
        )
        if noise_only:
            res = noise
        else:
            res = noise * eps + sample
        if l_flag:
            res_max = np.argmax(res, axis=-1)
            sample_max = np.argmax(sample, axis=-1)
            for idx in range(len(sample)):
                if res_max[idx] != sample_max[idx]:
                    t = res[idx][sample_max[idx]]
                    res[idx][sample_max[idx]] = res[idx][res_max[idx]]
                    res[idx][res_max[idx]] = t
            # res_max = np.argmax(res, axis=-1)
        return res

if __name__=='__main__':
    aa=listdir(path='savedModels\\fashion_10_32', fileType='', flagRe=False)[-5:]
    print(aa)