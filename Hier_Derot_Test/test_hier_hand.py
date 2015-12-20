__author__ = 'QiYE'
from xyz_base_wrist_derot_test_r0r1r2_conti import get_base_wrist_loc_err
from xyz_mid_derot_test_r0r1r2_conti import get_mid_loc_err
from xyz_top_derot_test_r0r1r2_conti import get_top_loc_err
from xyz_tip_derot_test_r0r1r2_conti import get_tip_loc_err


"""change the NUM_JNTS in src/constants.py to 6"""
setname='msrc'
xyz_jnt_save_path =   ['D:\\msrc_tmp\\jnt0_xyz.mat',
                       'D:\\msrc_tmp\\jnt1_xyz.mat',
                       'D:\\msrc_tmp\\jnt5_xyz.mat',
                       'D:\\msrc_tmp\\jnt9_xyz.mat',
                       'D:\\msrc_tmp\\jnt13_xyz.mat',
                       'D:\\msrc_tmp\\jnt17_xyz.mat']


get_base_wrist_loc_err(setname,xyz_jnt_save_path)

"""change the NUM_JNTS in src/constants.py to 1"""
''''change the path: xyz location of the palm center, file format can be npy or mat'''
setname='msrc'
file_format='mat'
prev_jnt_path =['D:\\msrc_tmp\\jnt1_xyz.mat',
              'D:\\msrc_tmp\\jnt5_xyz.mat',
              'D:\\msrc_tmp\\jnt9_xyz.mat',
              'D:\\msrc_tmp\\jnt13_xyz.mat',
              'D:\\msrc_tmp\\jnt17_xyz.mat']

xyz_jnt_save_path=['D:\\msrc_tmp\\jnt2_xyz.mat',
              'D:\\msrc_tmp\\jnt6_xyz.mat',
              'D:\\msrc_tmp\\jnt10_xyz.mat',
              'D:\\msrc_tmp\\jnt14_xyz.mat',
              'D:\\msrc_tmp\\jnt18_xyz.mat']

get_mid_loc_err(setname=setname,file_format=file_format,prev_jnt_path=prev_jnt_path,xyz_jnt_path=xyz_jnt_save_path)

"""change the NUM_JNTS in src/constants.py to 1"""
''''change the path: xyz location of the palm center, file format can be npy or mat'''
setname='msrc'
file_format='mat'
prev_jnt_path =['D:\\msrc_tmp\\jnt2_xyz.mat',
              'D:\\msrc_tmp\\jnt6_xyz.mat',
              'D:\\msrc_tmp\\jnt10_xyz.mat',
              'D:\\msrc_tmp\\jnt14_xyz.mat',
              'D:\\msrc_tmp\\jnt18_xyz.mat']

xyz_jnt_save_path=['D:\\msrc_tmp\\jnt3_xyz.mat',
              'D:\\msrc_tmp\\jnt7_xyz.mat',
              'D:\\msrc_tmp\\jnt11_xyz.mat',
              'D:\\msrc_tmp\\jnt15_xyz.mat',
              'D:\\msrc_tmp\\jnt19_xyz.mat']



get_top_loc_err(setname=setname,file_format=file_format,prev_jnt_path=prev_jnt_path,xyz_jnt_path=xyz_jnt_save_path)


"""change the NUM_JNTS in src/constants.py to 1"""
''''change the path: xyz location of the palm center, file format can be npy or mat'''
setname='msrc'
file_format='mat'
prev_jnt_path =['D:\\msrc_tmp\\jnt3_xyz.mat',
              'D:\\msrc_tmp\\jnt7_xyz.mat',
              'D:\\msrc_tmp\\jnt11_xyz.mat',
              'D:\\msrc_tmp\\jnt15_xyz.mat',
              'D:\\msrc_tmp\\jnt19_xyz.mat']

xyz_jnt_save_path=['D:\\msrc_tmp\\jnt4_xyz.mat',
              'D:\\msrc_tmp\\jnt8_xyz.mat',
              'D:\\msrc_tmp\\jnt12_xyz.mat',
              'D:\\msrc_tmp\\jnt16_xyz.mat',
              'D:\\msrc_tmp\\jnt20_xyz.mat']



get_tip_loc_err(setname=setname,file_format=file_format,prev_jnt_path=prev_jnt_path,xyz_jnt_path=xyz_jnt_save_path)