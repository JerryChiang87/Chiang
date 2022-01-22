import os



NET_NAME = 'resnet_v1_101' #

FIXED_BLOCKS = False

WEIGHT_DECAY = 0.001

num_classes = 15

global_pool = True #resnet做分类时需要设置成True

train_path = os.path.join('data', 'train')
test1_path = os.path.join('data', 'test1')
test2_path = os.path.join('data', 'test2')
cache_path = os.path.join('data', 'cache')
cache_rebuild = False
batch_size = 16
val_batch_size = 32
image_height = 796
image_width = 524
image_channels = 3

num_iters = 70000

decay_iters = 50000

lr = [0.01, 0.001]

beta1 = 0.9

beta2 = 0.999

momentum = 0.9

save_stp = 2000

batch_norm_scale = True

batch_norm_epsilon = 1e-5

batch_norm_decay = 0.997
