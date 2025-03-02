import torch
from torch import nn
from torch.optim import Adam
import torchvision as T
import torchvision.transforms as Transformer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import *
from datasets import *
import VGG
from argparse import ArgumentParser
import os


class PLModel(pl.LightningModule):
    def __init__(self, args = None, only_one_dataset = False):
        super(PLModel, self).__init__()
        if args is None:
            parser = ArgumentParser()
            parser = self.add_model_args(parser)
            self.args = parser.parse_args()
        else:
            self.args = args
        self.only_one_dataset = only_one_dataset
        self.save_hyperparameters()
        self.model = Model()
        vgg = VGG.vgg
        vgg.load_state_dict(torch.load(self.args.vgg_ckp_file))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.net = VGG.Net(vgg)
        self.L_loss = LaplacianRegularizer()
        self.epoch_indx = 0

    @staticmethod
    def add_model_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=4, help="batch size")
        parser.add_argument("--vgg_ckp_file", type=str, default="./checkpoints/vgg_normalised.pth")
        parser.add_argument("--content_path", type=str, default="/mnt/DataDisk/public/dataset/COCO/2017/images/train2017")
        parser.add_argument("--style_path", type=str, default="/mnt/DataDisk/public/dataset/COCO/2017/images/train2017")
        parser.add_argument("--lambda_c", type=float, default=0.5, help="coeff for loss c")
        parser.add_argument("--lambda_s", type=float, default=1, help = "coeff for loss s")
        parser.add_argument("--lambda_r", type=float, default=0.15, help = "coeff for loss r")
        parser.add_argument("--num_workers", type=int, default = 6)
        parser.add_argument("--img_size", type=int, default= 512)
        parser.add_argument("--lr", type=float, default= 1e-4, help = "learning rate")
        parser.add_argument("--val_content", type=str, default="./images/content_test/")
        parser.add_argument("--val_style", type=str, default="./images/style_test/")
        parser.add_argument("--log_training", type=int, default=False)
        parser.add_argument("--log_validation", type=int, default=True)
        parser.add_argument("--val_set_size", type=int, default=-1)
        parser.add_argument("--train_set_size", type=int, default=-1)
        parser.add_argument("--val_one_by_one", type=int, default=False)
        return parser

    def prepare_data(self):
        content_path = self.args.content_path
        style_path = self.args.style_path
        self.train_dataset = JBLDataset(content_path, style_path, img_size=self.args.img_size, set_size=self.args.train_set_size)
        self.val_dataset = JBLDataset(self.args.val_content, self.args.val_style, 
            img_size =self.args.img_size, one_by_one=self.args.val_one_by_one, 
            set_size = self.args.val_set_size,
            num_gpus = len(self.args.gpus.split(',')))
        self.train_loader = DataLoader(self.train_dataset, num_workers=self.args.num_workers,
            batch_size = self.args.batch_size*2, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size)
    
    def val_dataloader(self):
        if not hasattr(self, 'val_loader'):
            self.prepare_data()
        return self.val_loader
    
    def train_dataloader(self):
        if not hasattr(self, 'train_loader'):
            self.prepare_data()
        return self.train_loader

    @auto_move_data
    def forward(self, low_cont, cont_img, style_img, low_style):
        # input image tensors
        cont_feat = self.net.encode_with_intermediate(low_cont) # VGG 前4层的feature map
        style_feat = self.net.encode_with_intermediate(low_style) 
        coeffs,output = self.model(cont_img,cont_feat,style_feat)
        return coeffs,output

    def sample_image(self, vgg, model,batch, logname, img_idx):
        cont_img,low_cont,style_img,low_style = batch
        batch_size = cont_img.shape[0]
        model.eval()
        cont_feat = vgg.encode_with_intermediate(low_cont)
        style_feat = vgg.encode_with_intermediate(low_style)
        coeffs, output = model(cont_img, cont_feat, style_feat)

        cont = make_grid(cont_img, nrow=batch_size, normalize=True)
        style = make_grid(style_img, nrow=batch_size, normalize=True)
        out = make_grid(output, nrow=batch_size, normalize=True)

        image_grid = torch.cat((cont, style, out), 1)
        self.logger.experiment.add_image(logname, image_grid, img_idx)
        # save_image(image_grid, output_file + 'output'+str(epoch)+'.jpg', normalize=False)
        model.train()
        return image_grid

    def validation_step(self, batch, batch_idx):
        low_cont, cont_img, low_style, style_img = batch
        if self.args.log_validation:
            batch = [cont_img,low_cont,style_img,low_style]
            self.sample_image(self.net, self.model, batch, "val",  self.epoch_indx)
            pass
    
    def training_step(self, batch, batch_idx):
        if self.only_one_dataset:
            low_conts, cont_imgs = batch
            bs = low_conts.shape[0]//2
            low_cont = low_conts[:bs, :].contiguous()
            cont_img = cont_imgs[:bs, :].contiguous()
            low_style = low_conts[bs:bs*2, :].contiguous()
            style_img = cont_imgs[bs:bs*2, :].contiguous()
            del low_conts, cont_imgs
        else:
            low_cont, cont_img, low_style, style_img = batch
        
        coeffs, output = self(low_cont, cont_img, style_img, low_style) 
        loss_c,loss_s  = self.net.loss(output,cont_img,style_img)
        loss_r = self.L_loss(coeffs)
        total_loss = self.args.lambda_c * loss_c + self.args.lambda_s * loss_s + self.args.lambda_r * loss_r
        self.log("train_loss", total_loss)
        if self.args.log_training:
            sample_idx = batch_idx + self.epoch_indx * len(self.train_loader)
            if sample_idx % 30 == 0:
                batch = [cont_img,low_cont,style_img,low_style]
                self.sample_image(self.net, self.model, batch,'train', sample_idx)
        return total_loss

    def training_epoch_end(self, reslut_list):
        self.epoch_indx += 1
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.args.lr)

if __name__ == "__main__":
    # print("VISABLE", os.environ['CUDA_VISIBLE_DEVICES'])
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./log")
    parser.add_argument("--logname", type=str, default = "pl")
    parser.add_argument("--test", type=int, default=False)
    parser = PLModel.add_model_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.test:
        logger = False
        # model = PLModel(args) 
        model = PLModel.load_from_checkpoint("./log/pl_coco_f/version_4/checkpoints/last.ckpt")
        model.freeze()
        model.to('cuda:4')
        model.args = args
        val_loader = model.val_dataloader()
        for i, batch in enumerate(val_loader):
            print("run {}".format(i))
            low_cont,cont_img,low_style,style_img = batch
            coef, output = model(*batch)
            output = output.cpu()
            out = make_grid(output, normalize=True)
            cont = make_grid(cont_img, normalize=True)
            style = make_grid(style_img, normalize=True)
            # diff = make_grid(output - cont_img, normalize=False)
            # diff2 = torch.abs(diff).sum(dim=0).repeat(3, 1, 1)
            # image =torch.cat((cont, style, out, diff, diff2), 1)
            image =torch.cat((cont, style, out), 1)
            save_image(image, "./output/fusion_{}.png".format(i))
            print("saved {}".format(i))
    else:
        logger = TensorBoardLogger(args.logdir, args.logname, flush_secs=1)
        if args.style_path == args.content_path:
            only_one_dataset = True
        else:
            only_one_dataset = False

        model = PLModel(args, only_one_dataset=only_one_dataset)
        ckp_set = ModelCheckpoint(save_last=True, monitor="train_loss")
        trainer = pl.Trainer.from_argparse_args(args, 
            checkpoint_callback=ckp_set, 
            logger = logger,
            # distributed_backend = 'ddp',
            # gpus=[2,3,4,5,6]
            )
        trainer.fit(model)
