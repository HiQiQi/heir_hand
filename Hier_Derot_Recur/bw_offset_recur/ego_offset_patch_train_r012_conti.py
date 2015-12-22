__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.hier_test_files.CNN_Model import CNN_Model_multi3_conv1
from src.hier_test_files.Train import update_params,get_gradients,update_params2,set_params
import time
from src import constants
def train_model(setname, source_name,batch_size,jnt_idx,c1,h1_out_factor,h2_out_factor,lamda):

    model_info='uvd_bw%s_r012_egoff2'%jnt_idx[0]
    print model_info

    dataset = 'train'
    src_path = '../../../data/%s/hier_derot_recur/bw_offset/best/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    print 'source path',path

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi(path=path,is_shuffle=True,
                                                                                             jnt_idx=jnt_idx)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    num_channel=train_set_x0.shape[1]
    print 'n_train_batches', n_train_batches

    dataset = 'test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y=  load_data_multi(path=path,is_shuffle=True,
                                                                                             jnt_idx=jnt_idx)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

    model = CNN_Model_multi3_conv1(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
                             num_channel=num_channel,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 3,
                pool_c00= 2,
                c10= c1,
                kernel_c10= 3,
                pool_c10= 2,
                c20= c1,
                kernel_c20= 2,
                pool_c20= 1,

                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)

    cost = model.cost(Y)


    gamma = 0.0
    yita = 0.0000
    grads = get_gradients(model, cost)
    updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')

    # updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    # save_path = '../../data/%s/recursive/base_wrist/ego_best/'%setname
    # model_save_path = "%sparam_cost_uvd_bw_r012_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep810.npy"%save_path
    # set_params(model_save_path, model.params)
    # print 'initialize use jnt in the same layer'


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')


    n_epochs =1500
    epoch = 0
    test_cost=[]
    train_cost=[]

    done_looping=False
    save_path =  '../../../data/%s/hier_derot_recur/bw_offset/'%setname
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    model.save(path=save_path,c00=c1,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
               train_cost=train_cost,test_cost=test_cost)
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
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = train_model(x0,x1,x2,drop, y)
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        # for prm in model.params:
        #     p_v = prm.get_value()
        #     print numpy.mean(p_v)

        # if momentum.get_value() < 0.99:
        #     new_momentum = 1. - (1. - momentum.get_value()) * 0.99
        #     momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 0:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=train_cost,test_cost=test_cost)

        print 'test ', test_cost[-1],'  train', train_cost[-1]

    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=train_cost,test_cost=test_cost)


if __name__ == '__main__':
    # train_model(setname='msrc',
    #             source_name='_iter2_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500',
    #             lamda = 0.003,
    #             batch_size = 100,
    #             jnt_idx = [4],
    #                 c1=64,
    #             h1_out_factor=1,
    #             h2_out_factor=2)


    train_model(setname='nyu',
                source_name='_iter1_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020',
                lamda = 0.01,
                batch_size = 100,
                jnt_idx = [4],
                    c1=64,
                h1_out_factor=1,
                h2_out_factor=2)


    # train_model(setname='icvl',
    #             source_name='_recur1_patch_uvd_derot_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep2380',
    #             lamda = 0.001,
    #             batch_size = 100,
    #             jnt_idx = [5],
    #                 c1=64,
    #             h1_out_factor=2,
    #             h2_out_factor=2)