from tensorflow.python.keras.backend_config import epsilon
import zzxFunc
import numpy as np
import tensorflow as tf
from zzxFunc import normal_gradient, project, random_uniform
import zzxDataset
import os
import zzxConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input
from zzxFunc import random_uniform, normal_gradient, project
from pgd_attack import PGD_Attack
from at_train_setting import pgd_at_config, terrace_at_config
from adv_training import PGD_AT, Terrace_AT
# from member_infer import conf_score_infer
import member_infer

def adv_test(
        vm, x_benign, y_benign, 
        epsilon, steps, step_size,
        lp=np.inf, 
        logits_std=None, 
    ):
    # vm = Model(
    #     inputs=vm.input,
    #     outputs=vm.layers[0 - 2].output
    # )
    # benign_logits=vm.predict(x_benign)
    # target_logits=np.random.normal(0, logits_std, y_benign.shape)
    # target_labels=zzxFunc.softmax(target_logits)

    atk=PGD_Attack(
        victim_model=vm, 
        data_shape=x_benign.shape[1:],
        num_classes=y_benign.shape[1],
    )
    atk.config(
        epsilon=epsilon, steps=steps, 
        lp=lp, step_size=step_size, 
        shuffle=False
    )


    adv_data=atk.gen_adv_all(
        benign_data=x_benign, 
        benign_labels=y_benign
    )
    atk.test_adv(
        benign_data=x_benign, 
        benign_labels=y_benign,
        adv_data=adv_data,
        plot_name='1.png', 
        # plot_title='tmp='+str(tmp)+' steps='+str(steps)+' alpha='+str(alpha)
    )


if __name__=='__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass

    tf.random.set_seed(
        123
    )
    
    # dataset=zzxDataset.CIFAR100(standardization=False)
    dataset=zzxDataset.CIFAR10(standardization=False)
    # dataset=zzxDataset.Fashion(standardization=False)
    
    tmp=2
    # epoch=[10,20,20,30,30,40,40]#
    # epoch=[40,40,40,40,40,40]#
    epoch=[20,20,20,20,20,5]#
    
    for epoch in [130]:# , 50

        if dataset.name in['cifar10', 'cifar100', 'yale', 'svhn']:
            victim_model = zzxConv.zzxVGG16(
                dataset,
                build_dir=False
            )
            victim_model.setModel()
            victim_model.load_model(
                weights_path=r'savedModels//' + 'VGG16' +'_' + dataset.name+ '.h5'
                # weights_path=r'savedModels/' + dataset.name+ '_pgd_at.h5'
            )
        else:
            conv_layers_num = 5
            init_filters = 32
            victim_model = zzxConv.zzxCNN(
                dataset,
                build_dir=False
            )
            victim_model.setModel(
                conv_layers_num=conv_layers_num,
                filters=init_filters,
                kernel_size=(3,3)
            )
            victim_model.load_model(
                weights_path=r'savedModels//'+'CNN'+'_'+dataset.name +'_'+str(conv_layers_num) +'_'+str(init_filters)+'.h5'
                # weights_path=r'savedModels/' + dataset.name+ '_pgd_at.h5'
            )

        vm = victim_model.model

        pgd_at=PGD_AT(
            victim_model=vm, dataset=dataset,
            epoch=epoch, batch_size=128,#128
            save_path='savedModels/'
        )
        pgd_at.set_attack(
            epsilon=pgd_at_config[dataset.name]['epsilon'], 
            steps=pgd_at_config[dataset.name]['steps'], 
            lp=pgd_at_config[dataset.name]['lp'], 
            step_size=pgd_at_config[dataset.name]['step_size'],
            opt=pgd_at_config[dataset.name]['opt']
        )#lmb=0.9
        pgd_at.adv_test(part='train', data_num=30)
        pgd_at.adv_test(part='test', data_num=30)
        print()

        pgd_at.adv_fit(
            lr_schedule=pgd_at_config[dataset.name]['lr_schedule'],
            stage_init=2,
        )

        pgd_at.adv_test(part='train', data_num=30)
        pgd_at.adv_test(part='test', data_num=30)
        print()

        test_size, batch_size=1000, 100#500, 30#
        x_train, y_train=pgd_at.select_benign('train', data_num=test_size)
        x_test, y_test=pgd_at.select_benign('test', data_num=test_size)

        x_adv_train, x_adv_test= np.zeros((test_size,)+dataset.input_shape), np.zeros((test_size,)+dataset.input_shape)
        # y_adv_train, y_adv_test= np.zeros((test_size, dataset.num_classes)), np.zeros((test_size, dataset.num_classes))
        for id in range(int(test_size/batch_size)):
            x_adv_train[id*batch_size:min((id+1)*batch_size,test_size)]=pgd_at.test_attack.gen_adv_batch(
                benign_data=x_train[id*batch_size:min((id+1)*batch_size,test_size)], 
                benign_labels=y_train[id*batch_size:min((id+1)*batch_size,test_size)]
            )
            x_adv_test[id*batch_size:min((id+1)*batch_size,test_size)]=pgd_at.test_attack.gen_adv_batch(
                benign_data=x_test[id*batch_size:min((id+1)*batch_size,test_size)], 
                benign_labels=y_test[id*batch_size:min((id+1)*batch_size,test_size)]
            )

        z_train=pgd_at.victim_model.predict(x_train)
        z_test=pgd_at.victim_model.predict(x_test)

        z_adv_train=pgd_at.victim_model.predict(x_adv_train)
        z_adv_test=pgd_at.victim_model.predict(x_adv_test)
        
        # acc
        y_p = np.argmax(z_train, axis=1)
        print(round(np.sum(np.argmax(y_train, axis=1) == y_p) / len(y_p), 3), end=' ')
        
        y_p = np.argmax(z_adv_train, axis=1)
        print(round(np.sum(np.argmax(y_train, axis=1) == y_p) / len(y_p), 3), end=' ')
        
        y_p = np.argmax(z_test, axis=1)
        print(round(np.sum(np.argmax(y_test, axis=1) == y_p) / len(y_p), 3), end=' ')
        
        y_p = np.argmax(z_adv_test, axis=1)
        print(round(np.sum(np.argmax(y_test, axis=1) == y_p) / len(y_p), 3), end=' ')
        
        # normal mi
        metric_train=member_infer.conf_score(y_train, z_train)
        metric_test=member_infer.conf_score(y_test, z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=True), end=' ')
        
        metric_train=member_infer.entropy(z_train)
        metric_test=member_infer.entropy(z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False), end=' ')
        
        metric_train=member_infer.cross_entropy(y_train, z_train)
        metric_test=member_infer.cross_entropy(y_test, z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False), end=' ')
        
        metric_train=member_infer.m_entropy(y_train, z_train)
        metric_test=member_infer.m_entropy(y_test, z_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False), end=' ')
        
        # adv mi
        metric_train=member_infer.conf_score(y_train, z_adv_train)
        metric_test=member_infer.conf_score(y_test, z_adv_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=True), end=' ')
        
        metric_train=member_infer.entropy(z_adv_train)
        metric_test=member_infer.entropy(z_adv_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False), end=' ')
        
        metric_train=member_infer.cross_entropy(y_train, z_adv_train)
        metric_test=member_infer.cross_entropy(y_test, z_adv_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False), end=' ')
        
        metric_train=member_infer.m_entropy(y_train, z_adv_train)
        metric_test=member_infer.m_entropy(y_test, z_adv_test)
        print(member_infer.gap_infer(metric_train, metric_test, large_or_samll=False), end=' ')

    print(dataset.name)
