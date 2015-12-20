from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi_base_uvd_normalized,get_thresh_bw_jnts,load_patches
from src.hier_test_files.CNN_Model import CNN_Model_multi3


def train_model(setname, source_name,prev_jnt_name,batch_size,jnt_idx,patch_size,offset_depth_range,num_enlarge,h1_out_factor,h2_out_factor,lamda):


    print 'offset_depth_range ',offset_depth_range
    model_info='uvd_bw_r012_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info


    dataset = 'train'
    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    direct = '../../data/%s/final_xyz_uvd/'%setname
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_path,is_shuffle=True,
                                                                                             jnt_idx=jnt_idx,
                                                                                             patch_size=patch_size,patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=3/2.0)

    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    dataset = 'test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    direct = '../../data/%s/final_xyz_uvd/'%setname
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_path,is_shuffle=True,
                                                                                         jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32

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
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 5,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.sum_of_cost(Y,ratio=1.0,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)

    gamma = 0.0
    yita = 0.00005
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    # save_path = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/best/'
    # model_save_path = "%sparam_cost_base_wrist_r0r1r2_uvd_21jnts_derot_lg0_patch64_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep700.npy"%save_path
    # set_params(model_save_path, model.params)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],on_unused_input='ignore')


    n_epochs =1500
    epoch = 00
    test_cost=[]
    test_cost_loc=[]
    test_cost_rot=[]
    train_cost=[]
    train_cost_loc=[]
    train_cost_rot=[]

    done_looping=False
    save_path =    '../../data/%s/base_wrist_spacial/'%setname
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)
        test_cost_loc.append(cost_loc_nbatch/n_test_batches)
        test_cost_rot.append(cost_rot_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = train_model(x0,x1,x2,drop, y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        train_cost_loc.append(cost_loc_nbatch/n_train_batches)
        train_cost_rot.append(cost_rot_nbatch/n_train_batches)
        # if momentum.get_value() < 0.99:
        #     new_momentum = 1. - (1. - momentum.get_value()) * 0.99
        #     momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 20:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])

        print 'test ', test_cost[-1],test_cost_loc[-1],test_cost_rot[-1]
        print 'train', train_cost[-1],train_cost_loc[-1],train_cost_rot[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])



def train_model_spacial_tmp_patch56_555():

    batch_size = 100
    num_enlarge=0
    patch_size=56
    offset_depth_range=1.0
    print 'offset_depth_range ',offset_depth_range
    model_info='base_wrist_r012_uvd_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info, constants.OUT_DIM

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_patches('train_patch56_1.0')
    print 'train_patch56_1.0','test_patch56_1.0'
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=2)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_patches('test_patch56_1.0')
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32
    h1_out_factor=4
    h2_out_factor=16
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
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 5,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.sum_of_cost(Y,ratio=1,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)

    gamma = 0.0
    lamda = 0.01
    yita = 0.00005
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    save_path = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/best/'
    model_save_path = "%sparam_cost_base_wrist_r012_uvd_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep1000.npy"%save_path
    set_params(model_save_path, model.params)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],on_unused_input='ignore')


    n_epochs =2000
    epoch = 1000
    test_cost=[]
    test_cost_loc=[]
    test_cost_rot=[]
    train_cost=[]
    train_cost_loc=[]
    train_cost_rot=[]

    done_looping=False
    save_path ='../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/'
    model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
               train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)
        test_cost_loc.append(cost_loc_nbatch/n_test_batches)
        test_cost_rot.append(cost_rot_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = train_model(x0,x1,x2,drop, y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        train_cost_loc.append(cost_loc_nbatch/n_train_batches)
        train_cost_rot.append(cost_rot_nbatch/n_train_batches)
        # if momentum.get_value() < 0.99:
        #     new_momentum = 1. - (1. - momentum.get_value()) * 0.99
        #     momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 20:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])

        print 'test ', test_cost[-1],test_cost_loc[-1],test_cost_rot[-1]
        print 'train', train_cost[-1],train_cost_loc[-1],train_cost_rot[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])
def train_model_spacial_tmp_patch(patch_size,offset_depth_range,h1_out_factor,h2_out_factor,lamda):

    batch_size = 100
    num_enlarge=0
    print 'offset_depth_range ',offset_depth_range
    model_info='base_wrist_r0r1r2_uvd_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info, constants.OUT_DIM

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_patches('train_patch%d_1.0'%patch_size)
    print 'train_patch%d_1.0'%patch_size
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=2)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_patches('test_patch%d_1.0'%patch_size)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32

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
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 5,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.sum_of_cost(Y,ratio=1,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)

    gamma = 0.0
    yita = 0.00005
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    # save_path = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/best/'
    # model_save_path = "%sparam_cost_base_wrist_r0r1r2_uvd_21jnts_derot_lg0_patch64_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep700.npy"%save_path
    # set_params(model_save_path, model.params)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],on_unused_input='ignore')


    n_epochs =1500
    epoch = 00
    test_cost=[]
    test_cost_loc=[]
    test_cost_rot=[]
    train_cost=[]
    train_cost_loc=[]
    train_cost_rot=[]

    done_looping=False
    save_path =    '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/'
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)
        test_cost_loc.append(cost_loc_nbatch/n_test_batches)
        test_cost_rot.append(cost_rot_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = train_model(x0,x1,x2,drop, y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        train_cost_loc.append(cost_loc_nbatch/n_train_batches)
        train_cost_rot.append(cost_rot_nbatch/n_train_batches)
        # if momentum.get_value() < 0.99:
        #     new_momentum = 1. - (1. - momentum.get_value()) * 0.99
        #     momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 20:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])

        print 'test ', test_cost[-1],test_cost_loc[-1],test_cost_rot[-1]
        print 'train', train_cost[-1],train_cost_loc[-1],train_cost_rot[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])
def train_model_spacial_tmp_patch56_666():

    batch_size = 100
    num_enlarge=0
    patch_size=56
    offset_depth_range=1.0
    print 'offset_depth_range ',offset_depth_range
    model_info='base_wrist_r0r1r2_uvd_21jnts_derot_lg%d_patch%d_666'%(num_enlarge,patch_size)
    print model_info, constants.OUT_DIM

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_patches('train_patch56_1.0')
    print 'train_patch56_1.0','test_patch56_1.0'
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=2)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_patches('test_patch56_1.0')
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32
    h1_out_factor=4
    h2_out_factor=16
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
                kernel_c01= 4,
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 2,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.sum_of_cost(Y,ratio=1,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)

    gamma = 0.0
    lamda = 0.006
    yita = 0.00005
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    # save_path = '../../data/center/best/'
    # model_save_path = "%sparam_cost_center_r0r1r2_uvd_21jnts_c14_c28_gm0_lm800_yt0_ep50.npy"%save_path
    # set_params(model_save_path, model.params)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],on_unused_input='ignore')


    n_epochs =1000
    epoch = 0
    test_cost=[]
    test_cost_loc=[]
    test_cost_rot=[]
    train_cost=[]
    train_cost_loc=[]
    train_cost_rot=[]

    done_looping=False
    save_path =    '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/'
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)
        test_cost_loc.append(cost_loc_nbatch/n_test_batches)
        test_cost_rot.append(cost_rot_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = train_model(x0,x1,x2,drop, y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        train_cost_loc.append(cost_loc_nbatch/n_train_batches)
        train_cost_rot.append(cost_rot_nbatch/n_train_batches)
        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.99
            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 20:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])

        print 'test ', test_cost[-1],test_cost_loc[-1],test_cost_rot[-1]
        print 'train', train_cost[-1],train_cost_loc[-1],train_cost_rot[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])
def train_model_spacial_tmp_patch72_666():

    batch_size = 100
    num_enlarge=0
    patch_size=72
    offset_depth_range=1.0
    print 'offset_depth_range ',offset_depth_range
    model_info='base_wrist_r0r1r2_uvd_21jnts_derot_lg%d_patch%d_666'%(num_enlarge,patch_size)
    print model_info, constants.OUT_DIM

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_patches('train_patch56_1.0')
    print 'train_patch56_1.0','test_patch56_1.0'
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=2)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_patches('test_patch56_1.0')
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32
    h1_out_factor=4
    h2_out_factor=16
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
                kernel_c01= 8,
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 7,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 4,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.sum_of_cost(Y,ratio=1,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)

    gamma = 0.0
    lamda = 0.003
    yita = 0.00005
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    # save_path = '../../data/center/best/'
    # model_save_path = "%sparam_cost_center_r0r1r2_uvd_21jnts_c14_c28_gm0_lm800_yt0_ep50.npy"%save_path
    # set_params(model_save_path, model.params)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2],on_unused_input='ignore')


    n_epochs =1500
    epoch = 700
    test_cost=[]
    test_cost_loc=[]
    test_cost_rot=[]
    train_cost=[]
    train_cost_loc=[]
    train_cost_rot=[]

    done_looping=False
    save_path =    '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/'
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)
        test_cost_loc.append(cost_loc_nbatch/n_test_batches)
        test_cost_rot.append(cost_rot_nbatch/n_test_batches)

        cost_nbatch = 0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot = train_model(x0,x1,x2,drop, y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        train_cost_loc.append(cost_loc_nbatch/n_train_batches)
        train_cost_rot.append(cost_rot_nbatch/n_train_batches)
        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.99
            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 20:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])

        print 'test ', test_cost[-1],test_cost_loc[-1],test_cost_rot[-1]
        print 'train', train_cost[-1],train_cost_loc[-1],train_cost_rot[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=momentum.get_value()*10000,lamda=learning_rate.get_value()*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])
def train_model_spacial_tmp_patch64_dist_angle():

    batch_size = 100
    num_enlarge=0
    patch_size=64
    offset_depth_range=1.0
    print 'offset_depth_range ',offset_depth_range
    model_info='base_wrist_r0r1r2_uvd_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info, constants.OUT_DIM

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_patches('train_patch64_1.0')
    print 'train_patch64_1.0','test_patch64_1.0'
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=2)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_patches('test_patch64_1.0')
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32
    h1_out_factor=4
    h2_out_factor=16
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
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 5,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.sum_of_cost_3(Y,ratio=1,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)
    cost_3 = model.cost_angle(Y)
    gamma = 0.5
    lamda = 0.01
    yita = 0.00005
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.
    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    # grads = get_gradients(model, cost)
    # updates = update_params(model.params,grads,gamma=gamma,yita=yita,lamda=lamda)
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)

    # save_path = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/'
    # model_save_path = "%sparam_cost_base_wrist_r0r1r2_uvd_21jnts_derot_lg0_patch64_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep505.npy"%save_path
    # set_params(model_save_path, model.params)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2,cost_3],updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2,cost_3],on_unused_input='ignore')


    n_epochs =1500
    epoch = 0
    test_cost=[]
    test_cost_loc=[]
    test_cost_rot=[]
    test_cost_ang=[]
    train_cost=[]
    train_cost_loc=[]
    train_cost_rot=[]
    train_cost_ang=[]

    done_looping=False
    save_path =    '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/base_wrist_derot_spacial/'
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        cost_ang_nbatch=0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot,cost_ang = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_ang_nbatch+=cost_ang
            cost_nbatch+=cost_ij
        test_cost.append(cost_nbatch/n_test_batches)
        test_cost_loc.append(cost_loc_nbatch/n_test_batches)
        test_cost_rot.append(cost_rot_nbatch/n_test_batches)
        test_cost_ang.append(cost_ang_nbatch/n_test_batches)

        cost_nbatch = 0
        cost_loc_nbatch=0
        cost_rot_nbatch=0
        cost_ang_nbatch=0
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij,cost_loc,cost_rot,cost_ang = train_model(x0,x1,x2,drop, y)
            cost_loc_nbatch+=cost_loc
            cost_rot_nbatch+=cost_rot
            cost_ang_nbatch+=cost_ang
            cost_nbatch+=cost_ij

        train_cost.append(cost_nbatch/n_train_batches)
        train_cost_loc.append(cost_loc_nbatch/n_train_batches)
        train_cost_rot.append(cost_rot_nbatch/n_train_batches)
        train_cost_ang.append(cost_ang_nbatch/n_train_batches)

        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.99
            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # # adaption of learning rate
        # new_learning_rate = learning_rate.get_value() * 0.999
        # learning_rate.set_value(numpy.cast[theano.config.floatX](new_learning_rate))
        if epoch > 20:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
                           train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])

        print 'test ', test_cost[-1],test_cost_loc[-1],test_cost_rot[-1],test_cost_ang[-1]
        print 'train', train_cost[-1],train_cost_loc[-1],train_cost_rot[-1],train_cost_ang[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*10000,lamda=lamda*1000000,yita=yita*1000,epoch=epoch,
                   train_cost=[train_cost,train_cost_loc,train_cost_rot],test_cost=[test_cost,test_cost_loc,test_cost_rot])
if __name__ == '__main__':
    train_model(setname='icvl',
                source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                prev_jnt_name='_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1445',
                lamda = 0.01,
                batch_size = 100,
                jnt_idx = [0,1,5,9 ,13,17],
                num_enlarge=0,
                patch_size=56,
                offset_depth_range=0.6,
                h1_out_factor=2,
                h2_out_factor=4)

