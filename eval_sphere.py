import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mp
import time
from eval_best_attack import get_normal_vm, get_vm, get_terrace_vm
import zzxDataset
import os
import member_infer
import zzxFunc

# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(3)
# dataset=zzxDataset.CIFAR10(standardization=False)

def plot_dis(vm, part, tmp, sample_num, alpha):

    which_model=vm.name
    
    dis_sphere=np.load(get_filehead(which_model, part, tmp, alpha)+'dis_sphere.npy')
    label_sphere=np.load(get_filehead(which_model, part, tmp, alpha)+'label_sphere.npy')
    print(which_model+'_'+part+'_'+str(tmp))
    
    lo=np.argmin(np.reshape(dis_sphere[:,:,:],(-1,9)),axis=-1)
    ptb=dis_sphere[:,:,:]
    plt.hist(
        lo,
        np.arange(0, 9, 1),
        histtype='bar',
        rwidth=0.8,
        color='red',
        alpha=0.4
    )
    plt.savefig(fname='4.png')
    plt.clf()
    print(np.histogram(lo, bins=np.arange(0, 9, 1)))
    # print(label_sphere.shape)
    for idx in np.random.randint(0,500,sample_num):
        # print([round(ls, 3) for ls in label_sphere[0,0,idx]])
        # print(np.max(label_sphere[0,0,:], axis=-1))
        # idy=np.random.randint(0,9,1)[0]
        # print(np.argmax(label_sphere[idx,idy,:], axis=-1))
        print([round(ds, 3) for ds in dis_sphere[idx,0]])
        for idy in range(9):
            plt.plot(np.max(label_sphere[idx,idy,:], axis=-1), dis_sphere[idx,idy])
        plt.yticks(np.linspace(0,3,15))

        plt.savefig(fname='images/'+part+'_'+str(tmp)+'.png')
        plt.clf()
        time.sleep(0.5)


def print_Tmp_CfdScore_Ptb(vm, part, alpha, tmps, sec_num=9):

    which_model=vm.name
    print('Ptb '+which_model+'_'+part)

    print(' ', end='')
    label_sphere=np.load(get_filehead(which_model, part, 1, alpha)+'label_sphere.npy')
    for idx in range(sec_num):
        print(np.max(label_sphere[:,:,idx,:]), end=' ')
    print()

    for tmp in tmps:#0.2,
        print(tmp, end=' ')
        dis_sphere=np.load(get_filehead(which_model, part, tmp, alpha)+'dis_sphere.npy')

        for idx in range(sec_num):
            print(round(np.mean(dis_sphere[:,:,idx]), 3), end=' ')
        # print(dis_sphere[0:3,:,-1])
        print()

def print_Tmp_CfdScore_Acc(vm, part, alpha, tmps, acc_part='bng', sec_num=9, save_adv_prd=False):

    which_model=vm.name
    print('adv_acc '+which_model+'_'+part)

    print(' ', end='')
    label_sphere=np.load(get_filehead(which_model, part, 1, alpha)+'label_sphere.npy')
    for idx in range(sec_num):
        print(np.max(label_sphere[:,:,idx,:]), end=' ')
    print()

    
    num_class=dataset.num_classes

    target_num=num_class-1
    if num_class>10:
        target_num=5

    for tmp in tmps:#0.2, 
        print(tmp, end=' ')
        file_head=get_filehead(which_model, part, tmp, alpha)
        # dis_sphere=np.load(file_head+'dis_sphere.npy')
        adv_data=np.load(file_head+'adv_data.npy')
        # benign_data=np.load(file_head+'benign_data.npy')
        label_sphere=np.load(file_head+'label_sphere.npy')
        benign_labels=np.load(file_head+'benign_labels.npy')
        
        if save_adv_prd:
            adv_prd=np.zeros_like(label_sphere)
        else:
            adv_prd=np.load(file_head+'adv_prd.npy')
        
        for idx in range(sec_num):
            acc=[]
            for idy in range(target_num):
                
                if save_adv_prd:
                    adv_prd[:,idy,idx,:]=vm.predict(adv_data[:,idy,idx,:])
                
                adv_rlt=np.argmax(adv_prd[:,idy,idx,:], axis=1)
                tar_rlt=np.argmax(label_sphere[:,idy,idx,:], axis=1)
                benign_rlt=np.argmax(benign_labels, axis=1)
                
                if acc_part=='bng':
                    acc.append(1-round(np.sum(benign_rlt == adv_rlt) / len(adv_rlt),4))
                elif acc_part=='tar':
                    acc.append(round(np.sum(tar_rlt == adv_rlt) / len(adv_rlt), 4))
            
            print(round(np.mean(np.array(acc)), 4), end=' ')
        # print(dis_sphere[0:3,:,-1])
        print()
        if save_adv_prd:
            np.save(file_head+'adv_prd', adv_prd)

def evl_MI(vm, tmps, alpha, sample_num, class_num, sec_num=9, plot_flag=False, target_num=10):
        
    which_model=vm.name
    print('MI_acc '+which_model)
    
    # target_num=class_num-1
    # if class_num>10:
    #     target_num=5

    tmp_num=len(tmps)
    cfd_num=sec_num
    train_ptb=np.zeros((sample_num, target_num, cfd_num, tmp_num))
    test_ptb=np.zeros((sample_num, target_num, cfd_num, tmp_num))
    # train_adv_prd=np.zeros((sample_num, target_num, cfd_num, class_num, tmp_num))
    # test_adv_prd=np.zeros((sample_num, target_num, cfd_num, class_num, tmp_num))

    for idx in range(tmp_num):
        
        train_file_head=get_filehead(which_model, 'train', tmps[idx], alpha)
        test_file_head=get_filehead(which_model, 'test', tmps[idx], alpha)
        
        # for i in range(sample_num):
        #     for j in range(target_num):
        #         for k in range(cfd_num):
        #             if np.argmax(train_adv_prd[i, j, k, :, idx])==np.argmax(train_benign_labels[i]):
        #                 train_ptb[i, j, k, idx]=np.inf
        
        train_ptb[:, :, :, idx]=np.load(train_file_head+'dis_sphere.npy')
        train_adv_prd=np.load(train_file_head+'adv_prd.npy')
        train_benign_labels=np.load(train_file_head+'benign_labels.npy')
        bng_rlt=np.argmax(train_benign_labels, axis=1)
        
        for i in range(target_num):
            for j in range(cfd_num):
                adv_rlt=np.argmax(train_adv_prd[:, i, j, :],axis=1)
                adv_fail=np.where(adv_rlt==bng_rlt, 1, 0)
                fail_idx=np.where(adv_fail==1)[0]
                train_ptb[:, i, j, idx][fail_idx]=np.inf
        #         print(round(1-np.sum(adv_fail)/len(adv_fail), 3), end=' ')
        # print()

        test_ptb[:,:,:,idx]=np.load(test_file_head+'dis_sphere.npy')
        test_adv_prd=np.load(test_file_head+'adv_prd.npy')
        test_benign_labels=np.load(test_file_head+'benign_labels.npy')
        bng_rlt=np.argmax(test_benign_labels, axis=1)
        
        for i in range(target_num):
            for j in range(cfd_num):
                adv_rlt=np.argmax(test_adv_prd[:, i, j, :],axis=1)
                adv_fail=np.where(adv_rlt==bng_rlt, 1, 0)
                fail_idx=np.where(adv_fail==1)[0]
                test_ptb[:, i, j, idx][fail_idx]=np.inf
        
        #         print(round(1-np.sum(adv_fail)/len(adv_fail), 3), end=' ')
        # print()
        # print()

    train_ptb=np.transpose(train_ptb, (0, 2, 3, 1))
    test_ptb=np.transpose(test_ptb, (0, 2, 3, 1))

    # train_ptb=np.reshape(train_ptb, (sample_num, cfd_num, tmp_num, -1))
    # train_ptb=np.reshape(train_ptb, (sample_num, cfd_num, tmp_num, -1))
    # print(train_ptb.shape)
    # print(test_ptb.shape)
    
    print(' ', end='')
    label_sphere=np.load(get_filehead(which_model, 'train', 1, alpha)+'label_sphere.npy')
    for idx in range(sec_num):
        print(np.max(label_sphere[:,:,idx,:]), end=' ')
    print()
    
    for tmp_idx in range(tmp_num):

        print(tmps[tmp_idx], end=' ')
        
        for cfd_idx in range(cfd_num):
            mi_acc=member_infer.gap_infer(
                np.min(train_ptb[:, cfd_idx, tmp_idx, :], axis=1), 
                np.min(test_ptb[:, cfd_idx, tmp_idx, :], axis=1),
                large_or_samll=True,
                plot_flag=plot_flag
            )
            print(mi_acc, end=' ')
        print()

dataset=zzxDataset.CIFAR10(standardization=False)
# dataset=zzxDataset.CIFAR100(standardization=False)
# dataset=zzxDataset.MNIST(standardization=False)
# dataset=zzxDataset.Fashion(standardization=False)
# dataset=zzxDataset.Yale(standardization=False)

def get_filehead(which_model, part, tmp, alpha):
    return 'data/'+dataset.name+'_'+which_model+'_'+part+'_'+str(tmp)+'_'+str(alpha)+'_'#2_'

if __name__=='__main__':
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    

    normal_vm=get_normal_vm(dataset)
    pgd_vm=get_vm(model_name='pgd_at', model_path='savedModels/'+dataset.name+'_'+'pgd_at'+'.h5', dataset=dataset)

    # part='test'#'train'#
    # tmp=1 #[]
    # vm=terrace_vm_1 #
    alpha=0.01
    sec_num=4
    save_adv_prd=True#False#
    target_num=9#10#
    
    tmps=[0.5, 1, 3, 5]

    sample_num=1000
    class_num=dataset.num_classes

    for vm in [pgd_vm, normal_vm,]:#
        
        which_model=vm.name

        # plot_dis(which_model, part, tmp, 30, alpha)
        
        print_Tmp_CfdScore_Acc(
            vm, 'train', alpha, tmps, 
            acc_part='bng', sec_num=sec_num, 
            save_adv_prd=save_adv_prd
        )
        print()
        print_Tmp_CfdScore_Acc(
            vm, 'test', alpha, tmps, 
            acc_part='bng', sec_num=sec_num, 
            save_adv_prd=save_adv_prd
        )
        print()
        
        print_Tmp_CfdScore_Ptb(vm, 'train', alpha, tmps, sec_num=sec_num)
        print()
        print_Tmp_CfdScore_Ptb(vm, 'test', alpha, tmps, sec_num=sec_num)
        print()

        evl_MI(vm, tmps, alpha, sample_num=sample_num, class_num=class_num, sec_num=sec_num, target_num=target_num)
        print()
        
        train_head=get_filehead(vm.name, 'train', 1, alpha)
        x_train=np.load(train_head+'benign_data.npy')
        # x_train=np.load(train_head+'adv_data.npy')
        y_train=np.load(train_head+'benign_labels.npy')
        z_train=zzxFunc.softmax(vm.predict(x_train))
        
        test_head=get_filehead(vm.name, 'test', 1, alpha)
        x_test=np.load(test_head+'benign_data.npy')
        # x_test=np.load(test_head+'adv_data.npy')
        y_test=np.load(test_head+'benign_labels.npy')
        z_test=zzxFunc.softmax(vm.predict(x_test))

        metric_train=member_infer.conf_score(y_train, z_train)
        metric_test=member_infer.conf_score(y_test, z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=True))#, end=' '
        
        metric_train=member_infer.cross_entropy(y_train, z_train)
        metric_test=member_infer.cross_entropy(y_test, z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False))#, end=' '
        
        metric_train=member_infer.entropy(z_train)
        metric_test=member_infer.entropy(z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False))#, end=' '
        
        metric_train=member_infer.m_entropy(y_train, z_train)
        metric_test=member_infer.m_entropy(y_test, z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False))#, end=' '

        print()