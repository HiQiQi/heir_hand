__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import update_params,get_gradients,set_params,update_params2
import time
from src.utils import constants
def train_model(setname,dataset_path_prefix,source_name,lamda,c1,c2,h1_out_factor,h2_out_factor):

    # jnt_type='base' # jnt_type : base,mid, tip
    batch_size = 100
    model_info='whole_derot_21jnts_r012_conti'
    jnt_idx = range(0,21,1)
    print jnt_idx
    dataset = 'train'
    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi(path,jnt_idx,is_shuffle=True)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    dataset = 'test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(path,jnt_idx,is_shuffle=True)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

    model = CNN_Model_multi3(
        model_info=model_info,
        X0=X0,X1=X1,X2=X2,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 3,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= c1,
                kernel_c20= 3,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)

    cost = model.cost(Y)
    gamma = 0.0
    yita = 0.000

    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )
    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)
    #
    save_path = '%sdata/%s/whole_derot/'%(dataset_path_prefix,setname)
    model_save_path = "%sparam_cost_whole_derot_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm400_yt0_ep1305.npy"%save_path
    set_params(model_save_path, model.params)

    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)
    save_path =  '%sdata/%s/whole_derot/'%(dataset_path_prefix,setname)
    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')


    n_epochs =1800
    epoch = 1305
    test_cost=[]
    train_cost=[]
    done_looping=False
    drop =numpy.cast['int32'](0)
    print 'drop out ', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)

        cost_nbatch = 0
        # t0 = time.clock()
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = train_model(x0,x1,x2,drop, y)
            cost_nbatch+=cost_ij
        train_cost.append(cost_nbatch/n_train_batches)

        # if momentum.get_value() < 0.99:
        #     new_momentum = 1. - (1. - momentum.get_value()) * 0.985
        #     momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate


        if epoch > 10:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                           gamma=momentum.get_value()*1000,lamda=learning_rate.get_value()*100000,yita=yita*10000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)

        # t1 = time.clock()
        # print 'time', t1 - t0, 'momentum ', momentum.get_value(),'learning_rate ', learning_rate.get_value()
        print 'test', test_cost[-1], 'train', train_cost[-1]

    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                   gamma=momentum.get_value()*1000,lamda=learning_rate.get_value()*100000,yita=yita*10000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
if __name__ == '__main__':
    train_model(setname='icvl',
                dataset_path_prefix=constants.Data_Path,
                source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                lamda=0.004,c1=32,c2=64,h1_out_factor=8,h2_out_factor=32)
    # train_model(setname='msrc',source_name='_msrc_derot_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300', lamda=0.006,c1=32,c2=64,h1_out_factor=8,h2_out_factor=32)
    # train_model(setname='nyu',source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300', lamda=0.004,c1=32,c2=64,h1_out_factor=8,h2_out_factor=32)