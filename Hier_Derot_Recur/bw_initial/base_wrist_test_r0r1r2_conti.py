__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.Model.CNN_Model import CNN_Model_multi3,CNN_Model_multi3_conv3
from src.Model.Train import update_params,get_gradients,update_params2,set_params
import time
from src.utils import constants

def test_model(setname, source_name,prev_jnt_name,batch_size,jnt_idx,patch_size,offset_depth_range,num_enlarge,h1_out_factor,h2_out_factor,model_path,offset_save_path):


    print 'offset_depth_range ',offset_depth_range
    model_info='uvd_bw_r012_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info

    src_path = '../../data/%s/source/'%setname
    dataset = 'test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    direct = '../../data/%s/final_xyz_uvd/'%setname
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_path,is_shuffle=False,jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
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


    cost = model.cost(Y)

    save_path = '../../data/%s/base_wrist/'%setname
    model_save_path = "%s%s.npy"%(save_path,model_path)
    print model_save_path
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')

    cost_nbatch = 0
    uvd_norm = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        uvd_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij
    print 'cost', cost_nbatch/n_test_batches

    numpy.save("%s%s%s.npy"%(save_path,dataset,offset_save_path),uvd_norm)
def test_model_conv3_sequence(dataset,setname,dataset_path_prefix, source_name,batch_size,jnt_idx,
                     c1,c2,c3,h1_out_factor,h2_out_factor,model_path,offset_save_path):

    model_info='uvd_bw_r012_21jnts'
    print model_info
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(path,is_shuffle=False, jnt_idx=jnt_idx)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches



    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')



    model = CNN_Model_multi3_conv3(
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
                kernel_c01= 4,
                pool_c01= 2 ,
                c02= c3,
                kernel_c02= 3,
                pool_c02= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2 ,
                c12= c3,
                kernel_c12= 3,
                pool_c12= 2 ,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 5,
                pool_c21= 1 ,
                c22= c3,
                kernel_c22= 3,
                pool_c22= 1 ,

                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.cost(Y)



    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')

    for i in xrange(1000,2500,5):
        save_path =   '%sdata/%s/hier_derot_recur/bw_initial/'%(dataset_path_prefix,setname)

        model_save_path = "%sparam_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep%d.npy"%(save_path,i)
        print model_save_path
        set_params(model_save_path, model.params)
        cost_nbatch = 0
        uvd_norm = numpy.empty_like(test_set_y)
        for minibatch_index in xrange(n_test_batches):


            slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
            x0 = test_set_x0[slice_idx]
            x1 = test_set_x1[slice_idx]
            x2 = test_set_x2[slice_idx]
            y = test_set_y[slice_idx]

            cost_ij, uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            uvd_norm[slice_idx] = uvd_batch
            cost_nbatch+=cost_ij
        print 'cost', cost_nbatch/n_test_batches



def test_model_conv3(dataset,setname,dataset_path_prefix, source_name,batch_size,jnt_idx,
                     c1,c2,c3,h1_out_factor,h2_out_factor,model_path,offset_save_path):

    model_info='uvd_bw_r012_21jnts'
    print model_info
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(path,is_shuffle=False, jnt_idx=jnt_idx)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches



    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')



    model = CNN_Model_multi3_conv3(
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
                kernel_c01= 4,
                pool_c01= 2 ,
                c02= c3,
                kernel_c02= 3,
                pool_c02= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2 ,
                c12= c3,
                kernel_c12= 3,
                pool_c12= 2 ,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 5,
                pool_c21= 1 ,
                c22= c3,
                kernel_c22= 3,
                pool_c22= 1 ,

                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.cost(Y)

    save_path =   '%sdata/%s/hier_derot_recur/bw_initial/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_path)
    print model_save_path
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')

    cost_nbatch = 0
    uvd_norm = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        uvd_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij
    print 'cost', cost_nbatch/n_test_batches

    numpy.save("%s%s%s.npy"%(save_path,dataset,offset_save_path),uvd_norm)

if __name__ == '__main__':
    # train_model(setname='icvl',
    #             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             prev_jnt_name='_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1445',
    #             batch_size = 133,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             num_enlarge=0,
    #             patch_size=56,
    #             offset_depth_range=0.6,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_uvd_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1155',
    #             model_path='param_cost_uvd_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1155')

    # test_model_conv3(dataset='test',
    #                  setname='icvl',
    #                  dataset_path_prefix=constants.Data_Path,
    #         source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             batch_size = 133,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             c1=16,
    #             c2=32,
    #             c3=64,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep2380',
    #             model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep2380')

    test_model_conv3(dataset='train',
                     setname='icvl',
                     dataset_path_prefix=constants.Data_Path,
            source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                batch_size = 8,
                jnt_idx = [0,1,5,9 ,13,17],
                c1=16,
                c2=32,
                c3=64,
                h1_out_factor=2,
                h2_out_factor=4,
                offset_save_path='_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm5051_lm2000_yt0_ep2450',
                model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm5051_lm2000_yt0_ep2450')
    #
    # test_model_conv3(dataset='test',
    #                  setname='nyu',
    #         source_name='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             batch_size = 100,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             c1=16,
    #             c2=32,
    #             c3=48,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020',
    #             model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020')


    # test_model_conv3(dataset='test',
    #                  setname='msrc',
    #         source_name='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             batch_size = 100,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             c1=16,
    #             c2=32,
    #             c3=64,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500',
    #             model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500')



    # test_model(setname='nyu',
    #         source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #         prev_jnt_name='_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1000',
    #             batch_size = 100,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             num_enlarge=0,
    #             patch_size=56,
    #             offset_depth_range=0.8,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_uvd_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1160',
    #             model_path='param_cost_uvd_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1160')