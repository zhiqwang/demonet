model = dict(
    type='SSD',
    input_size=300,
    init_net=True,
    rgb_means=(103.94, 116.78, 123.68),
    growth_rate=32,
    block_config=[3, 4, 8, 6],
    num_init_features=32,
    bottleneck_width=[1, 2, 4, 4],
    drop_rate=0.05,
    p=0.6,
    anchor_config=dict(
        feature_maps=[38, 19, 10, 5, 3, 1],
        steps=[8, 16, 32, 64, 100, 300],
        min_ratio=20,
        max_ratio=90,
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        anchor_nums=[4, 6, 6, 6, 4, 4],
    ),
    num_classes=21,
    save_epochs=10,
    weights_save='weights/',
    pretained_model='weights/ssd.pth'
)

train_cfg = dict(
    cuda=True,
    per_batch_size=128,
    lr=5e-3,
    gamma=0.1,
    end_lr=5e-6,
    step_lr=[200000, 300000, 400000, 500000],
    print_epochs=10,
    num_workers=8,
)

test_cfg = dict(
    cuda=True,
    topk=0,
    iou=0.45,
    soft_nms=True,
    score_threshold=0.01,
    keep_per_class=200,
    save_folder='eval',
)

loss = dict(
    overlap_threshold=0.5,
    prior_for_matching=True,
    bkg_label=0,
    neg_mining=True,
    neg_pos_ratio=3,
    neg_overlap=0.5,
    encode_target=False,
)

dataset = dict(
    COCO=dict(
        train_sets=[('2017', 'train')],
        eval_sets=[('2017', 'val')],
        test_sets=[('2017', 'val')],
    )
)
