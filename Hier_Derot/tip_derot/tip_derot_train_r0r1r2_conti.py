from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi_tip_uvd_normalized
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import update_params2,set_params



def train_model(setname, dataset_path_prefix,source_name,prev_jnt_name,batch_size,jnt_idx,patch_size,offset_depth_range,c1,c2,h1_out_factor,h2_out_factor,lamda):

    model_info='offset_tip%d_r012_21jnts_derot'%jnt_idx[0]
    print model_info, constants.OUT_DIM


    dataset = 'train'
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    direct = '%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    print 'prev_jnt_path',prev_jnt_path
    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_tip_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,is_shuffle=True,
                                                                                            patch_size=patch_size, patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    dataset='test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    direct = '%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    print 'prev_jnt_path',prev_jnt_path
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_tip_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,is_shuffle=True,
                                                                                        patch_size=patch_size, patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
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
                kernel_c01= 4,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 1,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 2 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)
    cost = model.cost(Y)
    gamma = 0.0
    yita = 0.0000
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )

    # momentum implementation stolen from
    # http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert lamda >= 0. and lamda < 1.

    momentum =theano.shared(numpy.cast[theano.config.floatX](gamma), name='momentum')
    updates = update_params2(model,cost,momentum=momentum,learning_rate=learning_rate)


    print 'gamma_%f, lamda_%f,yita_%f'%(gamma, lamda,yita)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')

    # save_path =   '%sdata/%s/hier_derot/tip/'%(dataset_path_prefix,setname)
    # model_save_path = "%sparam_cost_offset_tip20_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm100_yt5_ep165.npy"%(save_path)
    # print model_save_path
    # set_params(model_save_path, model.params)

    n_epochs =1000
    epoch = 0
    test_cost=[]
    train_cost=[]
    done_looping=False
    save_path ='%sdata/%s/hier_derot/tip/'%(dataset_path_prefix,setname)
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
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

        if epoch > 10:
            if epoch%5 == 0:
                model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*100000,lamda=lamda*100000,yita=yita*100000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)

        # t1 = time.clock()
        print 'test ', test_cost[-1], 'train', train_cost[-1]
    if epoch == n_epochs:
        model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*100000,lamda=lamda*100000,yita=yita*100000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)

if __name__ == '__main__':
    mid_jnt_name=[]
    mid_jnt_name.append('_absuvd0_top3_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm100_yt0_ep205')
    mid_jnt_name.append('_absuvd0_top7_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep465')
    mid_jnt_name.append('_absuvd0_top11_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep465')
    mid_jnt_name.append('_absuvd0_top15_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep470')
    mid_jnt_name.append('_absuvd0_top19_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep465')
    idx =4
     # jnt_idx = [0,1,5,9 ,13,17]
    # jnt_idx = [2,6,10 ,14,18]
    # jnt_idx = [3,7,11,15,19]
    # jnt_idx = [4,8,12,16,20]
    train_model(setname='icvl',
                dataset_path_prefix=constants.Data_Path,
                source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                prev_jnt_name=mid_jnt_name[(idx-4)/4],
                lamda = 0.001,
                batch_size = 100,
                jnt_idx = [idx],
                patch_size=40,
                offset_depth_range=0.4,
                c1=14,
                c2=28,
                h1_out_factor=2,
                h2_out_factor=4)




    # nyu
    # mid_jnt_name=[]
    # mid_jnt_name.append('_absuvd0_top3_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep500')
    # mid_jnt_name.append('_absuvd0_top7_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep500')
    # mid_jnt_name.append('_absuvd0_top11_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep500')
    # mid_jnt_name.append('_absuvd0_top15_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep500')
    # mid_jnt_name.append('_absuvd0_top19_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep360')
