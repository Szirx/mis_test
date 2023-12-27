import matplotlib.pyplot as plt
import cv2


def plot_loss(history):
    plt.plot(history["val_loss"], label="val", marker="o")
    plt.plot(history["train_loss"], label="train", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def plot_score(history):
    plt.plot(history["train_miou"], label="train_mIoU", marker="*")
    plt.plot(history["val_miou"], label="val_mIoU", marker="*")
    plt.title("Score per epoch")
    plt.ylabel("mean IoU")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.plot(history["train_acc"], label="train_accuracy", marker="*")
    plt.plot(history["val_acc"], label="val_accuracy", marker="*")
    plt.title("Accuracy per epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def visualize(image_path, predicted_mask):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title('Picture')
    ax1.set_axis_off()

    ax2.imshow(predicted_mask)
    ax2.set_title('Predicted mask')
    ax2.set_axis_off()
