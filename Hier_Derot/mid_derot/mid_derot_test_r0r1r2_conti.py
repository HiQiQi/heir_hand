from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import load_data_multi_mid_uvd_normalized
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params


def test_model(dataset,setname, dataset_path_prefix,source_name,prev_jnt_name,batch_size,jnt_idx,patch_size,offset_depth_range,c1,c2,h1_out_factor,h2_out_factor,model_save_path,offset_save_path):

    num_enlarge=0
    print 'offset_depth_range ',offset_depth_range
    model_info='mid%d_offset_r012_21jnts_derot_lg%d_patch%d'%(jnt_idx[0],num_enlarge,patch_size)
    print model_info, constants.OUT_DIM
    src_path =  '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    direct ='%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    print 'prev_jnt_path',prev_jnt_path
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_mid_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,is_shuffle=False,
                                                                                        patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=offset_depth_range)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
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

    save_path =   '%sdata/%s/hier_derot/mid/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_save_path)
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
    #
    # test_model(dataset='train',setname='icvl',
    #            dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #             batch_size = 8,
    #             jnt_idx = [18],
    #             patch_size=40,
    #             offset_depth_range=0.4,
    #             c1=14,
    #             c2=28,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #                      model_save_path = 'param_cost_offset_mid18_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep605',
    #                      offset_save_path = '_offset_mid18_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep605' )

    test_model(dataset='test',setname='icvl',
               dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
                source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
                batch_size = 133,
                jnt_idx = [2],
                patch_size=40,
                offset_depth_range=0.4,
                c1=14,
                c2=28,
                h1_out_factor=2,
                h2_out_factor=4,
                         model_save_path = 'param_cost_offset_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm100_yt0_ep160',
                         offset_save_path = '_offset_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm100_yt0_ep160' )


    #
    # test_model(dataset='test',setname='icvl',
    #            dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #             batch_size = 133,
    #             jnt_idx = [6],
    #             patch_size=40,
    #             offset_depth_range=0.4,
    #             c1=14,
    #             c2=28,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #                      model_save_path = 'param_cost_offset_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep305',
    #                      offset_save_path = '_offset_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep305' )
    #
    # test_model(dataset='test',setname='icvl',
    #            dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #             batch_size = 133,
    #             jnt_idx = [10],
    #             patch_size=40,
    #             offset_depth_range=0.4,
    #             c1=14,
    #             c2=28,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #                      model_save_path = 'param_cost_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385',
    #                      offset_save_path = '_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385' )
    #



    # test_model(dataset='train',setname='nyu',
    #             source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             prev_jnt_name='_absuvd0_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep2000',
    #             batch_size = 8,
    #             jnt_idx = [2],
    #             patch_size=40,
    #             offset_depth_range=0.6,
    #             c1=14,
    #             c2=28,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #                      model_save_path = 'param_cost_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385',
    #                      offset_save_path = '_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385' )



    # test_model_tmp_patch(dataset='test',jnt_idx=[6],patch_size=40,  offset_depth_range=0.8,h1_out_factor=3,h2_out_factor=6,
    #                      model_save_path = 'param_cost_mid18_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep80',
    #                      offset_save_path = '_mid18_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep80')
    # test_model_tmp_patch(dataset='train',jnt_idx=[6],patch_size=32,  offset_depth_range=0.8,  h1_out_factor=2,h2_out_factor=4)
    # test(dataset='test',jnt_idx=[6],patch_size=40,  offset_depth_range=0.8,  h1_out_factor=3,h2_out_factor=6)
    # test(dataset='test',jnt_idx=[6],patch_size=32,  offset_depth_range=0.8,  h1_out_factor=2,h2_out_factor=4)