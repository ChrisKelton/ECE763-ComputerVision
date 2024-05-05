_base_ = [
    "../3rdparty/mmpretrain/configs/_base_/models/vit-base-p16.py",
    "../3rdparty/mmpretrain/configs/_base_/datasets/imagenet_bs64_pil_resize_autoaug.py",
    "../3rdparty/mmpretrain/configs/_base_/schedules/imagenet_bs4096_AdamW.py",
    "../3rdparty/mmpretrain/configs/_base_/default_runtime.py",
]

train_dataloader = dict(
    dataset=dict(
        data_root="/Users/twu19/Documents/Datasets/ILSVRC2015/Data/CLS-LOC",
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root="/Users/twu19/Documents/Datasets/ILSVRC2015/Data/CLS-LOC",
    ),
)

# model setting
model = dict(
    head=dict(hidden_dim=3072),
    train_cfg=dict(augments=dict(type="Mixup", alpha=0.2)),
)

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
