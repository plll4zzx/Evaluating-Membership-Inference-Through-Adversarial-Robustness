import zzxFunc
import numpy as np
import tensorflow as tf
from zzxFunc import normal_gradient, project, random_uniform
import zzxDataset
import os
import zzxConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input
import matplotlib.pyplot as plt
from distl_attack import Distill_Attack
from eval_best_attack import get_normal_vm, get_vm, get_terrace_vm

def one_hot2sphere(one_hot_labels, num_class=10, sec_num=10, target_num=None):
    benign_labels=np.argmax(one_hot_labels, axis=1)
    all_label=np.arange(0,num_class)
    if target_num is None:
        target_num=num_class
    
    sphere_labels=np.array([np.hstack((all_label[0:bl],all_label[bl+1:])) for bl in benign_labels])
    if num_class>10:
        rand_idx=np.random.choice(num_class-1, target_num)
        sphere_labels = sphere_labels[:, rand_idx]
    sphere_scores=np.zeros(sphere_labels.shape+(sec_num, num_class))
    
    for i in range(sphere_labels.shape[0]): 
         for j in range(sphere_labels.shape[1]): 
             for k in range(sec_num):
                conference_score=(k+2)/(sec_num+1)
                sphere_scores[i][j][k]=np.ones_like(sphere_scores[i][j][k])*(1-conference_score)/(num_class-1)
                sphere_scores[i][j][k][sphere_labels[i][j]]=conference_score
    # print(np.argmax(sphere_scores[2], axis=-1))
    return sphere_scores

if __name__=='__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    
    dataset=zzxDataset.CIFAR10(standardization=False)
    # dataset=zzxDataset.Fashion(standardization=False)
    # dataset=zzxDataset.CIFAR100(standardization=False)

    normal_vm=get_normal_vm(dataset)
    pgd_vm=get_vm(model_name='pgd_at', model_path='savedModels/'+dataset.name+'_'+'pgd_at'+'.h5', dataset=dataset)

    # logits_std=3
    adv_num=1000#300#
    batch_size=200#30#
    steps=500
    # alpha=0.01
    save_flag=True #False # 
    num_class=dataset.num_classes
    sec_num=4

    target_num=num_class-1
    if num_class>10:
        target_num=10
    else:
        target_num=num_class-1

    for alpha in [0.01]:#, 0.001,0.01
        for vm in [pgd_vm, normal_vm]:#
            for tmp in [1, 0.5, 3, 5]:#
                for part in ['test' ,'train']:#
                    
                    prj_name=dataset.name+'_'+vm.name+'_'+part+'_'+str(tmp)+'_'+str(alpha)
                    print(prj_name) #, end=': '+'_'+str(logits_std)
                    # if prj_name in ['yale_pgd_at_test_1_0.001',]:#[ 'terrace_at_1_test_0.2']
                    #     continue
                    dsta=Distill_Attack(
                        victim_model=vm, 
                        data_shape=dataset.input_shape,
                        num_classes=dataset.num_classes,
                    )
                    dsta.config(
                        steps=steps, alpha=alpha, tmp=tmp, 
                        lp=2, step_size=0.1, 
                        # shuffle=True, sta=499
                    )
                    
                    if part=='train':
                        batch_ids=np.random.choice(dataset.x_train.shape[0], size=adv_num)
                        benign_data=dataset.x_train[batch_ids]
                        benign_labels=dataset.y_train[batch_ids]
                        benign_logits=vm.predict(benign_data)
                    elif part=='test':
                        batch_ids=np.random.choice(dataset.x_test.shape[0], size=adv_num)
                        benign_data=dataset.x_test[batch_ids]
                        benign_labels=dataset.y_test[batch_ids]
                        benign_logits=vm.predict(benign_data)

                    label_sphere=one_hot2sphere(benign_labels, num_class, sec_num=sec_num, target_num=target_num)
                    dis_sphere=np.zeros((adv_num, target_num, sec_num))
                    adv_data=np.zeros((adv_num, target_num, sec_num)+dataset.input_shape)
                    
                    for j in range(sec_num):
                        for i in range(target_num):

                            adv_data[:,i,j,:,:,:] = dsta.gen_adv_all(
                                benign_data=benign_data, 
                                benign_labels=benign_labels,
                                target_labels=label_sphere[:,i,j,:],
                                batch_size=batch_size,
                            )

                            dis_sphere[:,i,j]=zzxFunc.cal_distance(benign_data, adv_data[:,i,j,:], lp=2)
                            
                        print(np.max(label_sphere[:,i,j,:]), end=' ')
                        print(round(np.mean(dis_sphere[:,:,j]), 3), end=' ')

                        tmp_adv=np.reshape(adv_data,(adv_num*(target_num),sec_num)+dataset.input_shape)
                        adv_rlt=np.argmax(vm.predict(tmp_adv[:,j,:]), axis=1)

                        tmp_label=np.reshape(label_sphere,(adv_num*(target_num),sec_num,num_class))
                        tar_rlt=np.argmax(tmp_label[:,j,:], axis=1)
                        # benign_rlt=np.argmax(benign_labels, axis=1)
                        
                        print(round(np.sum(tar_rlt == adv_rlt) / len(adv_rlt),4))#, end=' '
                        # print(round(np.sum(benign_rlt == adv_rlt) / len(adv_rlt),4))

                        # b_acc=[]
                        # t_acc=[]
                        # for idy in range(target_num):
                        #     adv_rlt=np.argmax(vm.predict(adv_data[:,i,idy,:]), axis=1)
                        #     tar_rlt=np.argmax(label_sphere[:,i,idy,:], axis=1)
                        #     benign_rlt=np.argmax(benign_labels, axis=1)
                        #     t_acc.append(round(np.sum(tar_rlt == adv_rlt) / len(adv_rlt),4))
                        #     b_acc.append(round(np.sum(benign_rlt == adv_rlt) / len(adv_rlt),4))
                        # print(round(np.mean(np.array(t_acc)), 4), end=' ')
                        # print(round(np.mean(np.array(b_acc)), 4))

                    print()

                    if save_flag:
                        file_head='data/'+prj_name+'_'#2_'
                        np.save(file_head+'adv_data', adv_data)
                        np.save(file_head+'benign_data', benign_data)
                        np.save(file_head+'benign_labels', benign_labels)
                        np.save(file_head+'dis_sphere', dis_sphere)
                        np.save(file_head+'label_sphere', label_sphere)
