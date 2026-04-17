from s3d_dataset import Structured3DDataset

train_dataset = Structured3DDataset(
    root_dir=r"E:\Study\code\LightRecon3D\data\Structured3D",
    split="train",
    train_ratio=0.9,
    image_size=(512, 512)
)

val_dataset = Structured3DDataset(
    root_dir=r"E:\Study\code\LightRecon3D\data\Structured3D",
    split="val",
    train_ratio=0.9,
    image_size=(512, 512)
)

print("train_dataset file:", train_dataset.__class__.__module__)
print("All scenes list:", train_dataset.all_scenes)
print("Train scenes list:", train_dataset.scenes)
print("Val scenes list:", val_dataset.scenes)
print("Train ratio:", train_dataset.train_ratio)
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))