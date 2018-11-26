from training_setting import configs


class Config():
    """Hyper Parameter configurations"""

    use_gpu = True

    # DRAGAN experiments
    e_lr = 1e-3
    d_lr = 1e-4
    g_lr = 1e-4
    decay_steps = 5000
    decay_rate = 1.0

    #######################
    # Architecture params #
    #######################
    batch_size = 32
    scale = 10.

    g_ksize = 5
    g_dim = 64
    g_use_bn = True

    e_ksize = 5
    e_dim = 128
    e_use_bn = True

    d_ksize = 5
    d_dim = 256
    d_use_bn = False
    d_use_ln = True
    d_input_norm = False

    z_dim = 512

    noise = 'default'  # 'multi_view' or 'default'
    use_normal_noise = True
    use_tanh = False
    use_auto_d_update = False
    encoder_weight_decay = 1e-3
    encoder_use_avg_pooling = False
    gan_loss_type = 'DCGAN'  # LSGAN

    ########################
    # Training/Loss params #
    ########################
    content_loss_weight = 10.
    clf_loss_weight = 0.
    adv_loss_weight = 0.
    pose_inv_loss_weight = 1.
    enc_content_loss_weight = 0.
    enc_clf_loss_weight = 0.
    gp = 'dragan'  # None, wgan, dragan
    img_loss_types = ['l1', 'l2']
    vox_inv_loss_types = {
        'l1': 1.,
        'l2': 1.
    }
    vox_inv_loss_weight = 1.
    optimizer_type = 'adam'  # 'adam' | 'rmsprop'

    max_d_acc = 1.  # 0.75 for PrGAN | 1. for default

    max_g_iters = 10  # Number of iterations of Generator training after which GAN flips to Discriminator mode
    max_d_iters = 40  # Number of iterations of Discriminator training after which GAN flips to Generator mode

    update_g = True  # whether update G during the training
    update_d = True  # whether update D during the training
    update_e = True  # whether update E during the training

    ####################
    # Projector params #
    ####################
    projection_typ = 'perspective'  # perspective | orthographic
    resolution = 32
    vox_size = 32
    distance = 100
    projection_temperature = 1.
    pooling_typ = 'exp'  # exp | max
    binarize_data = True
    # "X", "Y", "Z" for discrete case, and "XYZ" for cont. case,
    rotation_axis = 'XYZ'
    viewpoints = 5
    rx_range = [-20., 40.]
    ry_range = [0., 360.]
    rz_range = [180., 180.]

    #####################
    # Validation params #
    #####################
    max_validation_batches = None
    validation_interval = 200
    logging_interval = 10

    ##################
    # Dataset params #
    ##################
    category_settings = configs
