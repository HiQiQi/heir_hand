from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_r1r2_rotzdiscrete,load_data_r1r2_rotzdiscrete_icvl
from src.Model.CNN_Model import CNN_Model_multi2_softmax
from src.Model.Train import set_params


def train_model(setname,source_name):

    # jnt_type='base' # jnt_type : base,mid, tip
    batch_size = 100
    model_info = 'rot_r1r2_bin%d'% constants.Num_Class
    print model_info
    dataset = 'test'
    src_path = '../../data/%s/source/'%setname
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    test_set_x1,test_set_x2,test_set_y= load_data_r1r2_rotzdiscrete(path,setname,model_type='testing',batch_size=batch_size,is_shuffle=False)
    n_test_batches = test_set_x1.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    print 'n_test_batches', n_test_batches

    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.ivector('target')
    c1=4
    c2=8
    h1_out_factor=2
    h2_out_factor=4
    model = CNN_Model_multi2_softmax(model_info=model_info,X0=X1,X1=X2,
                      img_size_0 = img_size_1,
                      img_size_1=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 9,
                pool_c00= 2,
                c01= c2,
                kernel_c01= 9,
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
    err = model.layers[-1].errors(Y)
    save_path = '../../data/icvl/rot/'
    model_save_path = "%sparam_cost_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep105.npy"%save_path
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X1,X2,is_train,Y],
        outputs=[cost,err, model.layers[-1].p_y_given_x ],on_unused_input='ignore')

    cost_nbatch = 0
    err_nbatch = 0
    uvd_norm = numpy.empty((test_set_y.shape[0], constants.Num_Class))
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, err_batch, uvd_batch = test_model(x1,x2,numpy.cast['int32'](0),y)
        uvd_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij
        err_nbatch +=  err_batch
    print 'cost', cost_nbatch/n_test_batches
    print 'err', err_nbatch/n_test_batches
    numpy.save("%s%s_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep105.npy"%(save_path,dataset),uvd_norm)


def train_model_icvl(dataset,dataset_path_prefix,setname,source_name,batch_size,c1,c2,h1_out_factor,h2_out_factor,model_path,pred_rot_path):
    model_info = 'rot_r1r2_bin%d'% constants.Num_Class

    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    test_set_x1,test_set_x2,test_set_y= load_data_r1r2_rotzdiscrete_icvl(path,model_type='testing',batch_size=batch_size,is_shuffle=False)
    n_test_batches = test_set_x1.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches
    print test_set_y .shape

    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]


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
                kernel_c01= 9,
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
    err = model.layers[-1].errors(Y)
    save_path =  '%sdata/%s/hier_derot/rot/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_path)
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X1,X2,is_train,Y],
        outputs=[cost,err, model.layers[-1].p_y_given_x ],on_unused_input='ignore')

    cost_nbatch = 0
    err_nbatch = 0
    uvd_norm = numpy.empty((test_set_y.shape[0], constants.Num_Class))
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, err_batch, uvd_batch = test_model(x1,x2,numpy.cast['int32'](0),y)
        uvd_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij
        err_nbatch +=  err_batch
    print 'cost', cost_nbatch/n_test_batches
    print 'err', err_nbatch/n_test_batches
    numpy.save("%s%s%s.npy"%(save_path,dataset,pred_rot_path),uvd_norm)
if __name__ == '__main__':
    # train_model(setname='nyu')
    train_model_icvl(dataset='train',batch_size=8,
                     setname='icvl',dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
                source_name='_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                c1=4,c2=8,
                h1_out_factor=2,h2_out_factor=4,
                model_path='param_cost_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep135',
                pred_rot_path='_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep135')