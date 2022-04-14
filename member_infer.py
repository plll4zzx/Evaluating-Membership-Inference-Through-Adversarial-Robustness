
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    # -p*log(p)
    p=p+0.0001
    log_p=np.log(p)
    etry=p*log_p
    etry=np.sum(-etry,axis=1)
    return etry

def cross_entropy(p, q):
    # -p*log(q)
    q=q+0.0001
    log_q=np.log(q)
    ce=-p*log_q
    ce=np.sum(ce,axis=1)
    return ce

def m_entropy(p, q):
    
    # q=q+0.0001
    me=-(q*np.log(1-q+0.0001))
    y=np.argmax(p, axis=1)

    for id in range(len(me)):
        me[id][y[id]]=-(1-q[id][y[id]])*np.log(q[id][y[id]]+0.0001)
    
    me=np.sum(me, axis=1)
    return me

def conf_score(y, z):
    cs=np.zeros(len(z))
    y=np.argmax(y, axis=1)
    for id in range(len(z)):
        cs[id]=z[id][y[id]]
    return cs
    # return np.max(z,axis=1)

def gap_infer(train_metric, test_metric, large_or_samll, plot_flag=False):
    
    acc=0
    tr_acc=0
    te_acc=0
    gap=0
    for trs in train_metric:
        if large_or_samll:
            a=np.sum(np.where(train_metric>=trs, 1, 0))/len(train_metric)
            b=np.sum(np.where(test_metric<trs, 1, 0))/len(test_metric)
        else:
            a=np.sum(np.where(train_metric<=trs, 1, 0))/len(train_metric)
            b=np.sum(np.where(test_metric>trs, 1, 0))/len(test_metric)
        if (a+b)/2>acc:
            tr_acc=a
            te_acc=b
            acc=(tr_acc+te_acc)/2
            gap=trs
    # print(round(tr_acc, 3), end=' ')
    # print(round(te_acc, 3), end=' ')
    # print(gap)

    if plot_flag:
        lowG = min(train_metric.min(), test_metric.min())
        highG = max(test_metric.max(), train_metric.max())
        step = (highG - lowG) / 20
        plt.hist(
            train_metric,
            np.linspace(lowG, highG, 20),
            histtype='bar',
            rwidth=0.8,
            color='blue',
            alpha=0.4
        )
        plt.hist(
            test_metric,
            np.linspace(lowG, highG, 20),
            histtype='bar',
            rwidth=0.8,
            color='red',
            alpha=0.4
        )
        plt.savefig('1.png')
        plt.clf()

    return round(acc, 3)

if __name__=='__main__':

    # train_metric=np.array([6,7,8,9,10])
    # test_metric=np.array([1,2,3,4,5])
    # print(conf_score_infer(train_metric, test_metric))

    p=np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,0,0],
    ])

    q=np.array([
        [0.5, 0.2, 0.3],
        [0.9, 0.1, 0],
        [0.2,0.1,0.7],
        [0.98,0.01,0.01],
    ])

    print(entropy(p))
    print(entropy(q))
    print(cross_entropy(p,q))
    print(m_entropy(p,q))