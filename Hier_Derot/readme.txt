
***********************

For testing, use the file name with test
***************************

Directories

./prepare_data
./prepare_data/analyse_data.py 
"""check the uvd range of the hand so that to set the parameters of prepare_data.py to crop hand area from the original image and normalized the uvd locatioins of hand joints and the hand image itself.
"""

./prepare_data/prepare_data.py
"""crop hand area from the original image and normalized the uvd locatioins of hand joints and the hand image itself.
"""

./rot
"""predict the in-plane rotation of the hand"""

./utils/data_derot.py
"""derot the hand image and create new dataset with the prefix of 'derot' in the file name """

./center_derot
"""predict the center of the hand with deroted dataset. the center location is the mean value of joint 0 and joint 9"""

./base_wrist_derot_spacial
"""predict the base and wrist joint of the hand, i.e. joint 0, 1,5,9,13,17 with the constraints added in the cost funtion of cnn, which is the distance between joints.
"""

./base_wrist_derot
"""predict the base and wrist joint of the hand, i.e. joint 0, 1,5,9,13,17 without any constraint. The direct output is offset to the center uvd predicted by the model in ./center_derot. Transfer the offset uvd to absolute uvd and xyz by ./base_wrist_derot/error.py
"""

./mid_derot
"""predict the mid joints, i.e. joint 2,6,10,14,18. The direct output is offset to the base uvd predicted by the model in ./base_wrist_derot. Transfer the offset uvd to absolute uvd and xyz by ./mid_derot/error.py"""

./top_derot

"""predict the top joints, i.e. joint 3,7,11,15,19. T"""

./tip_derot
"""predict the mid joints, i.e. joint 4,8,12,16,20."""
