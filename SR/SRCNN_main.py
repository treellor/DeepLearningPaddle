"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.08.05
    Description	:
            SRCNN 模型训练
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import argparse
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader, Subset

from utils.data_read import ImageDatasetSingleToPair, OptType
from utils.utils import save_model, load_model, calc_psnr, AverageMeter, make_grid
from SRCNN_models import SRCNN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train(opt):
    # 创建文件夹
    save_folder_image = os.path.join(opt.save_folder, r"SRCNN/images")
    save_folder_model = os.path.join(opt.save_folder, r"SRCNN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    # 读取数据
    dataset = ImageDatasetSingleToPair(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, is_same_size=True,
                                       optType=OptType.RESIZE, scale_factor=opt.scale_factor)
    data_len = dataset.__len__()
    val_data_len = opt.batch_size
    val_set = Subset(dataset, list(range(val_data_len)))
    train_set = Subset(dataset, list(range(val_data_len, data_len)))

    val_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # 建立模型
    device = paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")

    srcnn = SRCNN().to(device)
    optimizer = optim.Adam(parameters=srcnn.parameters())
    criterion = nn.MSELoss().to(device)

    # 已经训练的 epoch数量
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_path, srcnn, optimizer)

    # 强制保存最后一个epoch
    n_epochs = opt.epochs
    # 评估参数
    train_loss_all, val_loss_all = [], []
    train_psnr_all, val_psnr_all = [], []

    for epoch in tqdm(range(trained_epoch + 1, trained_epoch + n_epochs + 1), desc=f'epoch'):
        epoch_train_loss = AverageMeter()
        epoch_train_psnr = AverageMeter()
        srcnn.train()
        for images_hl in train_dataloader:
            img_lr = images_hl["test"]._to(device)
            img_hr = images_hl["ref"]._to(device)

            img_gen = srcnn(img_lr)

            # srcnn.zero_grad()
            optimizer.clear_grad()

            train_loss = criterion(img_gen, img_hr)
            train_loss.backward(retain_graph=True)
            optimizer.step()

            train_psnr = calc_psnr(img_gen.detach(), img_hr)

            epoch_train_loss.update(train_loss.item(), len(img_hr))
            epoch_train_psnr.update(train_psnr.item(), len(img_hr))

        train_loss_all.append(epoch_train_loss.avg)
        train_psnr_all.append(epoch_train_psnr.avg)

        srcnn.eval()
        epoch_val_loss = AverageMeter()
        epoch_val_psnr = AverageMeter()
        with paddle.no_grad():
            for idx, datas_hl in enumerate(val_dataloader):
                image_l = datas_hl["test"]._to(device)
                image_h = datas_hl["ref"]._to(device)

                image_gen = srcnn(image_l)

                val_loss = criterion(image_gen, image_h)
                epoch_val_loss.update(val_loss.item(), len(image_l))
                val_psnr = calc_psnr(image_gen, image_h)
                epoch_val_psnr.update(val_psnr.item(), len(image_l))

                # save the result
                if epoch % opt.save_epoch_rate == 0 or (epoch == (trained_epoch + n_epochs)):
                    if idx == 0:
                        img_lr = make_grid(image_l, nrow=1, normalize=True)
                        img_hr = make_grid(image_h, nrow=1, normalize=True)
                        gen_hr = make_grid(image_gen, nrow=1, normalize=True)
                        img_grid = paddle.concat((img_hr, img_lr, gen_hr), axis=-1)
                        image_np = make_grid(img_grid, nrow=1, normalize=True).numpy() * 255
                        # Transpose to convert (C, H, W) to (H, W, C)
                        pil_image = Image.fromarray(image_np.astype('uint8').transpose(1, 2, 0))

                        pil_image.save(os.path.join(save_folder_image, f"epochs{epoch}.png"))

                        # 保存最新的参数和损失最小的参数
                        save_model(os.path.join(save_folder_model, f"epoch_{epoch}_model.pth"), srcnn, optimizer, epoch)

        val_loss_all.append(epoch_val_loss.avg)
        val_psnr_all.append(epoch_val_psnr.avg)

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_all, color='green', label='train loss')
    plt.plot(val_loss_all, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, r"loss.png"))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr_all, color='green', label='train PSNR dB')
    plt.plot(val_psnr_all, color='red', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, r"psnr.png"))
    plt.show()


def run(opt):
    save_folder_result = os.path.join(opt.save_folder, r"SRCNN/results")
    os.makedirs(save_folder_result, exist_ok=True)

    dataset = ImageDatasetSingleToPair(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, is_same_size=True,
                                       scale_factor=opt.scale_factor, optType=OptType.RESIZE, max_count=16)
    result_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=False)

    device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
    srcnn = SRCNN().to(device)
    load_model(opt.load_models_path, srcnn)

    srcnn.eval()
    for idx, datas_hl in tqdm(enumerate(result_loader), total=int(len(result_loader))):
        image_l = datas_hl["test"].to(device)
        image_h = datas_hl["ref"].to(device)
        image_gen = srcnn(image_l)

        img_lr = make_grid(image_l, nrow=1, normalize=True)
        img_hr = make_grid(image_h, nrow=1, normalize=True)
        gen_hr = make_grid(image_gen, nrow=1, normalize=True)
        img_grid = paddle.concat((img_hr, img_lr, gen_hr), axis=-1)
        image_np = make_grid(img_grid, nrow=1, normalize=True).numpy() * 255
        # Transpose to convert (C, H, W) to (H, W, C)
        pil_image = Image.fromarray(image_np.astype('uint8').transpose(1, 2, 0))
        pil_image.save(os.path.join(save_folder_result, f"epochs{epoch}.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='../data/T91', help='dataset path')
    parser.add_argument('--img_w', type=int, default=160, help='randomly cropped image width')
    parser.add_argument('--img_h', type=int, default=160, help='randomly cropped image height')
    parser.add_argument('--scale_factor', type=int, default=4, help='Image super-resolution coefficient')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path', type=str, default=r"./working/SRCNN/models/last_ckpt.pth",
                        help='load model path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--save_epoch_rate', type=int, default=100, help='How many epochs save once')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')

    args = parser.parse_args(args=[])

    return args


if __name__ == '__main__':

    para = parse_args()
    para.data_folder = '../data/T91'
    para.save_folder = r"./working/"
    para.img_w = 160
    para.img_h = 160
    para.scale_factor = 8
    para.batch_size = 4

    is_train = True

    if is_train:

        para.epochs = 2
        para.save_epoch_rate = 2
        para.load_models = False
        para.load_models_path = r"./working/SRCNN/models/epoch_50_model.pth"
        train(para)
    else:
        para.load_models = True
        para.load_models_path = r"./working/SRCNN/models/epoch_200_model.pth"
        run(para)
