from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_r1r2_rotzdiscrete,load_data_r1r2_rotzdiscrete_icvl
from src.Model.CNN_Model import CNN_Model_multi2_softmax
from src.Model.Train import update_params,get_gradients


def train_model(setname,source_name,dataset_path_prefix,
                lamda,
                c1,c2,
                h1_out_factor,h2_out_factor):

    # jnt_type='base' # jnt_type : base,mid, tip
    batch_size = 100
    model_info = 'rot_r1r2_bin%d'% constants.Num_Class
    print model_info
    dataset = 'train'
    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    # train_set_x1,train_set_x2,train_set_y= load_data_r1r2_rotzdiscrete_icvl(path)
    train_set_x1,train_set_x2,train_set_y= load_data_r1r2_rotzdiscrete(path,setname,model_type='training',batch_size=batch_size,is_shuffle=True)
    n_train_batches = train_set_x1.shape[0]/ batch_size
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches


    dataset = 'test'
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    test_set_x1,test_set_x2,test_set_y= load_data_r1r2_rotzdiscrete(path,setname,model_type='testing',batch_size=batch_size,is_shuffle=False)
    # test_set_x1,test_set_x2,test_set_y= load_data_r1r2_rotzdiscrete_icvl(path)
    n_test_batches = test_set_x1.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.ivector('target')

    model = CNN_Model_multi2_softmax(model_info=model_info,X0=X1,X1=X2,
                      img_size_0 = img_size_1,
                      img_size_1=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 9,
                pool_c00= 2,
                c01= c2,
                kernel_c01=9,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 7,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 4,
                pool_c11= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,

                batch_size = batch_size,
                p=0.5)

    cost = model.cost(Y)
    gamma = 0.0

    yita = 0.000
    grads = get_gradients(model, cost)
    updates = update_params(model.params,grads,gamma=gamma,lamda=lamda,yita=yita)

    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)
    # save_path = '../../data/rot/'
    # model_save_path = "%sparam_cost_rot_r1r2_discrete_c14_c28_h14_h28_gm0_lm0_yt0_ep100.npy"%save_path
    # set_params(model_save_path, model.params)
    train_model = theano.function(inputs=[X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')


    n_epochs =600
    epoch = 00
    test_cost=[]
    train_cost=[]
    done_looping=False
    save_path = '%sdata/%s/hier_derot/rot/'%(dataset_path_prefix,setname)
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = test_model(x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = train_model(x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij
        train_cost.append(cost_nbatch/n_train_batches)

        if epoch > 10:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10,lamda=lamda*100,yita=yita*1000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
        print 'test', test_cost[-1], 'train ', train_cost[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10,lamda=lamda*100,yita=yita*1000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
def train_model_icvl(setname,source_name,dataset_path_prefix,
                lamda,
                c1,c2,
                h1_out_factor,h2_out_factor):

    # jnt_type='base' # jnt_type : base,mid, tip
    batch_size = 100
    model_info = 'rot_r1r2_bin%d'% constants.Num_Class
    print model_info
    dataset = 'train'
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    train_set_x1,train_set_x2,train_set_y= load_data_r1r2_rotzdiscrete_icvl(path,model_type='training',batch_size=batch_size,is_shuffle=True)
    # train_set_x1,train_set_x2,train_set_y= load_data_r1r2_rotzdiscrete(path,setname)
    n_train_batches = train_set_x1.shape[0]/ batch_size
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches
    print train_set_y .shape

    dataset = 'test'
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    # test_set_x1,test_set_x2,test_set_y= load_data_r1r2_rotzdiscrete(path,setname)
    test_set_x1,test_set_x2,test_set_y= load_data_r1r2_rotzdiscrete_icvl(path,model_type='testing',batch_size=batch_size,is_shuffle=True)
    n_test_batches = test_set_x1.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches
    print test_set_y .shape

    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.ivector('target')

    model = CNN_Model_multi2_softmax(model_info=model_info,X0=X1,X1=X2,
                      img_size_0 = img_size_1,
                      img_size_1=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 9,
                pool_c00= 2,
                c01= c2,
                kernel_c01=9,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 7,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 4,
                pool_c11= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,

                batch_size = batch_size,
                p=0.5)

    cost = model.cost(Y)
    gamma = 0.0
    yita = 0.000
    grads = get_gradients(model, cost)
    updates = update_params(model.params,grads,gamma=gamma,lamda=lamda,yita=yita)

    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)
    # save_path = '../../data/rot/'
    # model_save_path = "%sparam_cost_rot_r1r2_discrete_c14_c28_h14_h28_gm0_lm0_yt0_ep100.npy"%save_path
    # set_params(model_save_path, model.params)
    train_model = theano.function(inputs=[X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')


    n_epochs =600
    epoch = 00
    test_cost=[]
    train_cost=[]
    done_looping=False
    save_path = '%sdata/%s/hier_derot/rot/'%(dataset_path_prefix,setname)
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = test_model(x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = train_model(x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij
        train_cost.append(cost_nbatch/n_train_batches)

        if epoch > 10:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10,lamda=lamda*10000,yita=yita*1000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
        print 'test', test_cost[-1], 'train ', train_cost[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10,lamda=lamda*100,yita=yita*100000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
if __name__ == '__main__':

    train_model_icvl(setname='icvl',dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
                source_name='_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                lamda=0.003,
                c1=4,c2=8,
                h1_out_factor=2,h2_out_factor=4)
    # train_model(setname='nyu',
    #             source_name='_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             lamda=0.01,
    #             c1=4,c2=8,
    #             h1_out_factor=2,h2_out_factor=4)