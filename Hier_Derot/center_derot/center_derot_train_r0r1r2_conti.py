from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi_center_continuous
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import update_params2,set_params


def train_model(setname, dataset_path_prefix,source_name,batch_size,lamda,c1,c2,h1_out_factor,h2_out_factor):

    model_info='center_r0r1r2_uvd'
    print model_info, constants.OUT_DIM

    dataset = 'train'
    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    print path
    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_center_continuous(path,is_shuffle=True)

    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    dataset = 'test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_center_continuous(path,is_shuffle=True)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

    model = CNN_Model_multi3(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
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
    yita = 0.00000
    save_path =  '%sdata/%s/hier_derot/center/'%(dataset_path_prefix,setname)
    model_save_path = "%sparam_cost_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep765.npy"%save_path
    set_params(model_save_path, model.params)
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.

    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')


    n_epochs =2500
    epoch =765
    test_cost=[]
    train_cost=[]
    done_looping=False
    save_path = '%sdata/%s/hier_derot/center/'%(dataset_path_prefix,setname)
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                   gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
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
        #     new_momentum = 1. - (1. - momentum.get_value()) * 0.99
        #     momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # if learning_rate.get_value()>0.00005:
        #     new_learning_rate = learning_rate.get_value() * 0.99
        #     learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        # print 'momentum ', momentum.get_value(),'learning_rate ', learning_rate.get_value()

        if epoch > 0:
            if epoch%5 == 0:

                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                           gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
        print 'test ', test_cost[-1], 'train', train_cost[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                   gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)

if __name__ == '__main__':
    train_model(setname='icvl',
                lamda = 0.000300,
                dataset_path_prefix=constants.Data_Path,
                source_name='_icvl_derot2_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                batch_size = 100,
                c1=16,
                c2=32,
                h1_out_factor=6,
                h2_out_factor=16)

    # train_model(setname='nyu',
    #             dataset_path_prefix=constants.Data_Path,
    #             lamda = 0.0003,
    #             source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             batch_size = 100,
    #             c1=16,
    #             c2=32,
    #             h1_out_factor=6,
    #             h2_out_factor=16,
    #             )