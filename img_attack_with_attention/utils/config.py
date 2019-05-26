from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'


class Config:
    # data
    voc_data_dir = '/home/liangsiyuan/data/VOCdevkit/VOC2007'
    # voc_data_dir = '/home/xingxing/liangsiyuan/data/VOCdevkit/VOC2007'
    # voc_data_dir = '/home/xlsy/Desktop/Datasets/VOC2007'
    # voc_data_dir = '/home/xlsy/Desktop/Datasets/cat_VOC'
    voc_test_data_dir = '/home/xlsy/Desktop/experiments/target0_0.00005_20/VOC2007'
    min_size = 300  # image resize
    max_size = 300


    # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.0
    roi_sigma = 1.0

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'attention+img_10_20'  # visdom env
    port = 8000
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = True # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 1000
    # model
    load_path = '/home/liangsiyuan/code/weights/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth'
    # load_path = '/home/xlsy/Desktop/fasterrcnn_img_0.701.pth'
    # load_path = '/home/xingxing/liangsiyuan/code/weights/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth'
    load_attacker = '/home/xlsy/Desktop/experiments/target0_0.00005_20/weights/attack_10240645.path'
    # load_path = '/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_02050841_13'

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16-caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
