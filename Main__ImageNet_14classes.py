import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import time
from utils_swin import AverageMeter, initialize_logger, save_checkpoint, record_loss
import torchvision
from torchvision import transforms
import shutil
# from swin1 import *
# from our_swin import *
# from our_swin_dense_no_transi_out2_Layernm import *
# from our_swin_dense_no_transi import *
# from swin_Den4KVStage3q_sota import *
# from models_mae_224_14 import *
# from swin_v2_224 import *
# from timm import create_model as creat
# from swin1 import *
# from our_swin import *
# from our_swin_dense_no_transi_out2_Layernm import *
# from our_swin_dense_no_transi import *
# from swin_Den4KVStage3q import *
# from GSNet_steel_withtransition import *
# from Swin_original import *
# from Swin_S import *
# from Swin_B import *
# from densenet import *
# from swin1 import *
# from GSNet_cifar import *
# from our_swin import *
# from our_swin_dense_no_transi_out2 import *
# from our_swin_dense_no_transi import *
# import input_data
# from cnn_finetune import make_model
# from efficientnet_pytorch import EfficientNet
# from Resnext50 import Resnext
# from ViT import ViT
from GSNet import *
import timm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,0,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'



parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--batchSize", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=666, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=0.9, help="decay power")
parser.add_argument("--max_iter", type=float, default=400000, help="max_iter")
parser.add_argument("--outf", type=str, default="./Results/ImageNet_14classes_mae/", help='path log files')
# parser.add_argument("--outf", type=str, default="./Results/ImageNet_14classes_EfficientNet-b0/", help='path log files')
opt = parser.parse_args()



def main():
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+'); log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir); print('save_path is already')

    torch.backends.cudnn.benchmark = True
    shutil.copyfile(os.path.basename(__file__), opt.outf + os.path.basename(__file__))

    print("\nloading dataset ...")
    dataset = tra.load_data('./datasets/ILSVRC-2012/train_14classes/', is_train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, sampler=None, num_workers=8, pin_memory=True, shuffle=True)
    dataset_test = tra.load_data('./datasets/ILSVRC-2012/val_14classes/', is_train=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=None, num_workers=8, pin_memory=True, shuffle=True)
    print("Train:%d" % (len(data_loader)))
    print("Validation set samples: ", len(data_loader_test))
    opt.max_iter = opt.end_epoch * len(data_loader)
    print("\nbuilding models_baseline ...")
    # model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), growth_rate=32, block_config=(6, 12, 24, 16),
    #              num_init_features=64, bn_size=4, drop_rate=0)

    # model = torchvision.models.resnet50(pretrained=True, num_classes=14)
    # model = torchvision.models.__dict__['resnext101_32x8d'](pretrained=True)
    # model = torchvision.models.__dict__['resnext50_32x4d'](pretrained=True)
    # model = torchvision.models.__dict__['resnet50'](pretrained=True)
    # channel_in = model.fc.in_features        
    # model.fc = nn.Linear(channel_in, 14)
    # model = torchvision.models.__dict__['densenet201'](pretrained=False)
    # channel_in = model.classifier.in_features  
    # model.classifier = nn.Linear(channel_in, 14)
    # model = densenet161()
    # model = torchvision.models.__dict__['vgg19'](pretrained=True)
    # model.classifier._modules['6'] = nn.Linear(4096, 14)
    # model = make_model(model_name='inception_v3', num_classes=14, pretrained=True)
    # model=EfficientNet.from_name('efficientnet-b0')
    # model = EfficientNet.from_pretrained('efficientnet-b0')
    # resume_file = os.path.join(os.path.join(opt.outf), 'efficientnet-b0-355c32eb.pth') 
    # if resume_file:
    #     if os.path.isfile(resume_file):
    #         print("=> loading checkpoint '{}'".format(resume_file))
    #         checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
    #         # start_epoch = checkpoint['epoch']
    #         # iteration = checkpoint['iter']
    #         # model.load_state_dict(checkpoint['model'], strict = False)
    #         model.load_state_dict(checkpoint, strict = False)
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    # num_ftrs = model._fc.in_features
    # model._fc = nn.Linear(num_ftrs, 14)
    # model = Resnext()
    # model = ViT(image_size=224,
    #             patch_size=32,
    #             num_classes=10,
    #             dim=1024,
    #             depth=6,
    #             heads=16,
    #             mlp_dim=2048,
    #             dropout=0.1,
    #             emb_dropout=0.1)
    # model = creat('vit_large_patch16_224', pretrained=True, num_classes=14)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), growth_rate=32,
                            block_config=(6, 12, 24, 16),
                            num_init_features=64, bn_size=4, drop_rate=0)
    # model = SwinTransformer_original()
    # model = SwinTransformer()
    # model = SwinTransformerV2()
    # model = MaskedAutoencoderViT(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # model = nn.DataParallel(model, device_ids=[0,1])

  '''
  re-implement of ConvNext, CoAtNet, EVA-02, and MaxViT
  '''
    # model = timm.create_model('convnext_xxlarge.clip_laion2b_soup_ft_in1k', pretrained=False, num_classes=14)
    criterion_train = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        model.cuda();
        criterion_train.cuda();
    print('Model is already and parameters number is ', sum(param.numel() for param in model.parameters()))

    start_epoch = 0;
    iteration = 0;
    record_acc = 0.5

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    resume_file = os.path.join(os.path.join(opt.outf), 'mae_finetuned_vit_base.pth')
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
            # start_epoch = checkpoint['epoch']
            # iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['model'], strict=False)
            # model.load_state_dict(checkpoint, strict = False)
            # optimizer.load_state_dict(checkpoint['optimizer'])


    for epoch in range(start_epoch + 1, opt.end_epoch):
        start_time = time.time()
        # train
        train_loss, iteration, lr = tra.train(data_loader, model, criterion_train, optimizer, epoch, iteration,
                                              opt.init_lr, opt.decay_power)
        acc = tra.validate(data_loader_test, model)  # 调用validate
        if abs(acc - record_acc) < 0.001 or acc > record_acc or epoch % 50 == 0: 
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)  
            if acc > record_acc:
                record_acc = acc
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f acc: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, acc))
        record_loss1(loss_csv, epoch, iteration, epoch_time, lr, train_loss, acc)
        logger.info("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate: %.9f, Train Loss: %.9f acc: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, acc))  





class tra(nn.Module):

    def train(self, train_loader, model, criterion, optimizer, epoch, iteration, init_lr,
              decay_power): 
        model.train()  
        losses = AverageMeter()  
        for i, (images, labels) in enumerate(train_loader): 
            labels = labels.cuda();
            labels = Variable(labels)
            images = images.cuda();
            images = Variable(images) 
            lr = tra.poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=opt.max_iter,
                                       power=decay_power) 
            iteration = iteration + 1  # iter+1
            output = model(images) 
            # output = model.forward_encoder(images, mask_ratio=0.75)
            loss = criterion(output, labels) 
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()  
            losses.update(loss.data)  
            print('[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f' % (
            epoch, i, len(train_loader), iteration, lr, losses.avg))  
        return losses.avg, iteration, lr

    def validate(self, val_loader, model):
        model.eval() 
        ac = 0
        idx_to_class = {0: '10.0', 1: '10.5', 2: '11.0', 3: '11.5', 4: '12.0', 5: '12.5', 6: '13.0', 7: '6.5', 8: '7.0',
                        9: '7.5', 10: '8.0', 11: '8.5', 12: '9.0', 13: '9.5'}
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda();
            target = target.cuda()
            with torch.no_grad():  
                output = model(input)  
                # output = model.forward_encoder(input, mask_ratio=0.75)
                _, pred = output.topk(1, 1, True, True)
                pred = pred.flatten().cpu().numpy()
                target = target.cpu().numpy()
                target = torch.tensor([float(idx_to_class[target[i]]) for i in range(len(target))])
                pre_label = torch.tensor([float(idx_to_class[pred[i]]) for i in range(len(pred))])
                acc0, acc0_5 = tra.accuracy(pre_label, target)
                ac += (0.4 * acc0 + 0.6 * acc0_5) / len(val_loader)
        return ac

    def accuracy(self, output, target):
        pre_label = output.cuda()
        gt_label = target.cuda()
        acc0 = torch.le(abs(pre_label - gt_label), 0)
        acc0_5 = torch.le(abs(pre_label - gt_label), 0.5)
        return acc0, acc0_5

    def poly_lr_scheduler(self, optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):  # 学习率按iter次数下降
        if iteraion % lr_decay_iter or iteraion > max_iter:
            return optimizer
        lr = init_lr * (1 - iteraion / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def load_data(self, dir, is_train):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            dataset = torchvision.datasets.ImageFolder(dir, transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),  # 随机形变压缩裁剪至224*224
                transforms.ToTensor(), normalize]))
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            dataset = torchvision.datasets.ImageFolder(dir, transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),  # 压缩至256*256 中心裁剪224*224大小的验证图像
                transforms.ToTensor(), normalize]))
            sampler = torch.utils.data.SequentialSampler(dataset)
        return dataset, sampler




if __name__ == '__main__':
    if torch.cuda.is_available():
        tra = tra().cuda()
    main()  
    print(torch.__version__)  
