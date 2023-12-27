import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm.notebook import tqdm
import sys
print(sys.path)
from src.train.metrics import mean_iou, pixel_accuracy
from src.train.datasets import TrainDataset
from src.train.datasetSplitter import create_df
import configs.config as cfg
import configs.transformConfig as tfCfg
import mlflow
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    train_losses, test_losses = [], []
    train_iou, val_iou = [], []
    train_acc, val_acc = [], []
    lrs = []
    min_loss = np.inf

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = batch

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mean_iou(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = batch

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mean_iou(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print(
                    "Loss Decreasing.. {:.3f} >> {:.3f} ".format(
                        min_loss, (test_loss / len(val_loader))
                    )
                )
                min_loss = test_loss / len(val_loader)
                print("saving model...")
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss_function_state_dict": criterion,
                    },
                    "model_for_train.pt",
                )
                torch.save(model, "torch_best.model.pt")

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print(
                "Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                "Time: {:.2f}m".format((time.time() - since) / 60),
            )

    history = {
        "train_loss": train_losses,
        "val_loss": test_losses,
        "train_mean_iou": train_iou,
        "val_mean_iou": val_iou,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lrs": lrs,
    }
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history


if __name__ == "__main__":

    df = create_df()
    X_train, X_val = train_test_split(df['id'].values, test_size=0.2, random_state=19)
    # datasets
    train_set = TrainDataset(cfg.IMAGE_PATH, cfg.MASK_PATH, X_train, cfg.MEAN, cfg.STD, tfCfg.t_train)
    val_set = TrainDataset(cfg.IMAGE_PATH, cfg.MASK_PATH, X_val, cfg.MEAN, cfg.STD, tfCfg.t_val)

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=True)

    smp_model = smp.Unet(
        cfg.PRETRAINED_MODEL,
        encoder_weights="imagenet",
        classes=cfg.NUM_CLASSES,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        smp_model.parameters(), lr=cfg.MAX_LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.WEIGHT_DECAY,
        epochs=cfg.EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    with mlflow.start_run():
        mlflow.pytorch.autolog()
        fit(
            epochs=cfg.EPOCHS,
            model=smp_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        mlflow.end_run()
