import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

transform = albu.Compose([
    # albu.OneOf([
    #     # albu.RGBShift(p=0.5),
    #     albu.ChannelShuffle(p=0.5)
    # ], p=0.5),
    albu.OneOf([
        albu.RandomBrightness(p=0.5),
        albu.RandomContrast(p=0.5),
        # albu.Sharpen(p=0.5, alpha=(0.3, 0.9), lightness=(0.75, 1.5)),
        # albu.GaussianBlur(blur_limit=(1, 3), p=0.5)
    ], p=0.5),
    albu.Affine(translate_px={"x": (-10, 10), "y": (-10, 10)}, rotate=(1, 5), p=0.5),
    # albu.OneOf([
    #     albu.RandomRotate90(p=0.5),
    #     albu.Flip(p=0.5)
    # ], p=0.5),
    # albu.ToFloat(p=1.0),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1.0, always_apply=True)
])
albu.save(
    transform,
    filepath='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/abnormal_detect/cfg/fast_transformer.yaml',
    data_format='yaml'
)