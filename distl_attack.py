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

# tf.config.experimental_run_functions_eagerly(True)

class Attack():

    def __init__(
        self, 
        victim_model, 
        data_shape,
        num_classes,
    ):
        self.victim_model = victim_model
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.first_test=True

    def gen_adv_all(
        self,  
        benign_data,
        benign_labels,
        target_labels=None,
        batch_size=100,
    ):
        adv_num=benign_data.shape[0]
        batch_num=int(np.ceil(adv_num/batch_size))
        adv_data=np.zeros_like(benign_data)
        
        # self.adv_data = tf.Variable(tf.random.uniform((batch_size,)+benign_data.shape[1:], -1,1, dtype='float32'), trainable=True)

        for idx in range(batch_num):

            # b_data=tf.convert_to_tensor(
            #     benign_data[idx*batch_size: min(adv_num, (idx+1)*batch_size)]
            # )
            # self.adv_data = tf.random.uniform(b_data.shape, -1,1, dtype='float32')
            # self.adv_data = tf.Variable(self.adv_data)

            if target_labels is not None: #min(adv_num, (idx+1)*batch_size)
                adv_data[idx*batch_size: min(adv_num, (idx+1)*batch_size)]=self.gen_adv_batch(
                    benign_data[idx*batch_size: min(adv_num, (idx+1)*batch_size)], 
                    benign_labels[idx*batch_size: min(adv_num, (idx+1)*batch_size)],
                    target_labels[idx*batch_size: min(adv_num, (idx+1)*batch_size)],
                )
                # self.adv_data.numpy()
            else:
                adv_data[idx*batch_size: min(adv_num, (idx+1)*batch_size)]=self.gen_adv_batch(
                    benign_data[idx*batch_size: min(adv_num, (idx+1)*batch_size)], 
                    benign_labels[idx*batch_size: min(adv_num, (idx+1)*batch_size)],
                )
                # self.adv_data.numpy()
        
        # if target_labels is not None:
        #     return adv_data, target_labels
        # else:
        return adv_data

    def test_adv(
        self, 
        benign_data,
        benign_labels,
        adv_data,
        target_labels=None,
        plot_flag=True, 
        plot_name='1.png', 
        plot_title='',
        batch_idx=None,
    ):
        if plot_flag:
            zzxFunc.plot_imgs(
                adv_data,
                img_path=plot_name,
                r=5, c=5,
                img_show_flag=False,
                img_save_flag=True,
                figsize=None,
                randomFlag=True
            )
            plt.clf()

            if self.logits_flag:
                tmp_labels=np.max(zzxFunc.softmax(self.victim_model.predict(benign_data)), axis=1)
            else:
                tmp_labels=np.max(self.victim_model.predict(benign_data), axis=1)
            plt.hist(
                tmp_labels,
                np.linspace(0,1,10),
                histtype='bar',
                rwidth=0.8,
                color='blue',
                alpha=0.4
            )
        
            if self.logits_flag:
                tmp_labels=np.max(zzxFunc.softmax(self.victim_model.predict(adv_data)), axis=1)
            else:
                tmp_labels=np.max(self.victim_model.predict(adv_data), axis=1)
            plt.hist(
                tmp_labels,
                np.linspace(0,1,10),
                histtype='bar',
                rwidth=0.8,
                color='red',
                alpha=0.4
            )
            plt.title(plot_title)
            plt.savefig('4.png')

        if self.first_test:
            self.first_test=False
            print(plot_title)
            test_y = np.argmax(self.victim_model.predict(benign_data), axis=1)
            print('test_benign:', end=' ')
            print(round(np.sum(np.argmax(benign_labels, axis=1) == test_y) / len(test_y),4))

            if target_labels is not None:
                print('test_Adv episilon test_target')
            else:
                print('test_Adv episilon')
        
        if batch_idx is not None:
            print(batch_idx, end=': ')

        adv_y = np.argmax(self.victim_model.predict(adv_data), axis=1)
        # print('test_Adv:', end='')
        test_adv_acc=round(np.sum(np.argmax(benign_labels, axis=1) == adv_y) / len(adv_y),4)
        print(test_adv_acc, end=' ')
        
        # print('episilon:', end='')
        ptb_budget=zzxFunc.cal_distance(benign_data, adv_data, lp=self.lp)
        ptb_budget_scale=round(np.mean(ptb_budget),3)
        print(ptb_budget_scale, end=' ')
        # print(round(np.min(ptb_budget),3), end=' ')
        # print(round(np.max(ptb_budget),3), end=' ')
        
        if target_labels is not None:
            # print('test_Adv_target:', end='')
            print(round(np.sum(np.argmax(target_labels, axis=1) == adv_y) / len(adv_y),4))
        else:
            print()
        return test_adv_acc, ptb_budget_scale

class Distill_Attack(Attack):

    def config(
        self, steps, alpha, tmp, lp, 
        step_size, target_flag=True, logits_flag=True, 
        shuffle=False, sta=50
    ):
        self.steps=steps
        self.alpha=alpha
        self.tmp=tmp
        self.lp=lp
        self.step_size=step_size
        self.target_flag=target_flag
        self.logits_flag=logits_flag
        self.shuffle=shuffle
        self.sta=sta
        self.first_test=True
    
    def fine_config(self, tmp, alpha):
        self.tmp=tmp
        self.alpha=alpha

    def trans_benign_logits(self, benign_logits):
        ...

    def find_best_adv(
        self, 
        benign_data, benign_labels,
        target_labels,
        steps, lp, step_size,
        alpha_scale=[10, 1, 0.1, 0.01, 0.001], 
        tmp_scale=[0.2,0.3,1,3,5],
        target_flag=True, logits_flag=True, 
        shuffle=False, sta=50
    ):
        self.steps=steps
        self.lp=lp
        self.step_size=step_size
        self.target_flag=target_flag
        self.logits_flag=logits_flag
        self.shuffle=shuffle
        self.sta=sta
        self.first_test=False
        
        adv_data={}
        for alpha in alpha_scale:
            adv_data[alpha]={}
            for tmp in tmp_scale:
                print(str(alpha)+'_'+str(tmp)+' ', end='')
                adv_data[alpha][tmp]={}
                self.fine_config(tmp=tmp, alpha=alpha)
                adv_data[alpha][tmp]['data']=self.gen_adv_all(
                    benign_data,
                    benign_labels,
                    target_labels,
                    batch_size=100,
                )
                adv_data[alpha][tmp]['adv_acc'], adv_data[alpha][tmp]['adv_budget']=self.test_adv(
                    benign_data,
                    benign_labels,
                    adv_data[alpha][tmp]['data'],
                    target_labels=target_labels,
                    plot_flag=False,
                )
        
        print()
        best_alpha=0
        best_tmp=0
        best_adv_acc=0.1
        for alpha in alpha_scale:
            for tmp in tmp_scale:
                if adv_data[alpha][tmp]['adv_acc']<best_adv_acc:
                    best_alpha=alpha
                    best_tmp=tmp
        # return adv_data[best_alpha][best_tmp]['data'], adv_data[best_alpha][best_tmp]['adv_acc'], adv_data[best_alpha][best_tmp]['adv_budget']


        # return adv_data

    def gen_adv_batch(self, benign_data, benign_labels, target_labels):
        # t_logits = target_labels

        # acc=[]
        # acc_step=[]
        # tf.reduce_sum

        # adv_data = np.random.uniform(-1,1,b_data.shape).astype('float32')
        benign_data=tf.convert_to_tensor(benign_data)
        target_labels=tf.convert_to_tensor(target_labels)
        adv_data = tf.random.uniform(benign_data.shape, -1,1, dtype='float32')
        adv_data = tf.Variable(adv_data)
        ce = tf.keras.losses.CategoricalCrossentropy()
        mse = tf.keras.losses.MeanSquaredError()
        opt = tf.keras.optimizers.Adam(learning_rate=self.step_size)
        
        @tf.function
        def attack_step(benign_data, target_labels, adv_data, opt):#

            with tf.GradientTape() as ce_tape, tf.GradientTape() as mse_tape:
                    
                ce_tape.watch(adv_data)
                mse_tape.watch(adv_data)
                
                tmp_adv = 0.5*(tf.tanh(adv_data)+1)
                tmp_ptbs = tmp_adv-benign_data
                prediction = self.victim_model(tmp_adv)
                
                distl_p = tf.keras.backend.softmax(prediction/self.tmp)
                
                loss_ce = ce(target_labels, distl_p)
                loss_mse = mse(tf.zeros_like(adv_data), tmp_ptbs)
            
            gradient_ce = ce_tape.gradient(loss_ce, adv_data)
            gradient_mse = mse_tape.gradient(loss_mse, adv_data)
            gradient=self.alpha*gradient_ce+gradient_mse

            opt.apply_gradients(zip([gradient], [adv_data]))

        for idx in range(self.steps):

            attack_step(benign_data, target_labels, adv_data, opt)

            if self.shuffle==True and idx%self.sta==0:
                tmp_adv = 0.5*(tf.tanh(adv_data)+1).numpy()
                # tmp_adv=tf.clip_by_value(adv_data+b_data, 0, 1).numpy()
                # print(idx, end='; ')
                self.test_adv(
                    benign_data=benign_data, 
                    benign_labels=benign_labels,
                    adv_data=tmp_adv,
                    target_labels=target_labels,
                    plot_name='2.png', 
                    plot_title='tmp='+str(self.tmp)+' steps='+str(self.steps)+' alpha='+str(self.alpha),
                    batch_idx=idx,
                )

        adv_data = 0.5*(tf.tanh(adv_data)+1)
        # adv_data=tf.clip_by_value(adv_data+b_data, 0, 1)
        # adv_data = tf.convert_to_tensor(adv_data)
        return adv_data.numpy()

if __name__=='__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])
    except:
        pass
    # tf.random.set_seed(
    #     123
    # )
    
    dataset=zzxDataset.CIFAR10(standardization=False)

    victim_model=zzxConv.zzxVGG16(
        dataset,
        build_dir=False,
    )
    victim_model.setModel()
    victim_model.load_model(
        # weights_path=r'savedModels//' + 'VGG16' +'_' + dataset.name+ '.h5'
        weights_path=r'cifar10_pgd_at_50.h5'
    )

    vm = Model(
        inputs=victim_model.model.input,
        outputs=victim_model.model.layers[0 - 2].output,
        name='victim_model'
    )

    # x_input = Input(shape=dataset.input_shape)
    # tmp_m = Model(
    #     inputs=victim_model.model.input,
    #     outputs=victim_model.model.layers[0 - 2].output
    # )
    # x_out = tmp_m(x_input)
    # x_out = Activation('softmax')(x_out/2)
    # vm = Model(
    #     inputs=x_input,
    #     outputs=x_out
    # )
    # vm.summary()
    # vm.load_weights('pgd_at.h5')

    dsta=Distill_Attack(
        victim_model=vm, 
        data_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
    )

    adv_num=1000
    batch_ids=np.random.choice(dataset.x_train.shape[0], size=adv_num)
    benign_data=dataset.x_train[batch_ids]
    benign_labels=dataset.y_train[batch_ids]
    benign_logits=vm.predict(benign_data)

    # # adv_dict={}
    # for alpha in [1, 0.1]:#, 0.01, 0.001
    #     # adv_dict[str(alpha)]={}
    #     for tmp in [1,3]:#,5,7
    #         # adv_dict[str(alpha)][str(tmp)]={}
    #         for logits_std in [1, 1.5]:#, 2, 3, 5, 10
    #             # adv_dict[alpha][tmp][logits_std]={}
    logits_std=3
    print(logits_std)
    steps=500
    alpha=0.01
    tmp=0.3

    target_logits=np.random.normal(0, logits_std, benign_labels.shape)
    target_labels=zzxFunc.softmax(target_logits)

    dsta.config(
        steps=steps, alpha=alpha, tmp=tmp, 
        lp=2, step_size=0.1, 
        # shuffle=True, sta=50
    )
    adv_data=dsta.gen_adv_all(
        benign_data=benign_data, 
        benign_labels=benign_labels,
        target_labels=target_labels
    )
    dsta.test_adv(
        benign_data=benign_data, 
        benign_labels=benign_labels,
        adv_data=adv_data,
        target_labels=target_labels,
        plot_name='1.png', 
        plot_title='tmp='+str(tmp)+' steps='+str(steps)+' alpha='+str(alpha)
    ) 
    print()
    
