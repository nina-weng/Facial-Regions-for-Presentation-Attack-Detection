import torch
import torchvision
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.extmath import softmax
from datetime import datetime
import numpy as np
import torch.backends.cudnn as cudnn
import os

parser = argparse.ArgumentParser(description='Training with feature-level-fusion resnet18. \
Dataset frame-based Casia-Face-AntiSpoofing PAD.')
parser.add_argument('--train-batch', default=4, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--size', default='256', choices=['154','140','256','192','combine'])
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--num-epochs', type=int, default=10, help="number of epochs")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--rec-dir', default='../../results/scores_rec/region_fusion', type=str,
                    help="score recording file path dir")
parser.add_argument('--face-regions', default='[forehead,face_ISOV]', type=str,
                    help="list-like string, the list of face regions")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()



# feature level fusion resnet18
class FLF_resnet18(nn.Module):
    def __init__(self, sub_net_num=2):
        super(FLF_resnet18, self).__init__()
        self.sub_net_num = sub_net_num
        net_list = []
        for i in range(self.sub_net_num):
            net_this =torchvision.models.resnet18(pretrained=True)
            net_list.append(net_this)
        self.net_list = nn.ModuleList(net_list)
        self.bn = nn.BatchNorm1d(self.sub_net_num*1000)
        self.fc = nn.Linear(self.sub_net_num*1000,2)

    def forward(self, x):
        assert(self.sub_net_num == x.shape[1])
        feature_list=[]
        for i in range(self.sub_net_num):
            feature_list.append(self.net_list[i](x[:,i]))
        feature=torch.cat(feature_list,1)
        out =self.bn(feature)
        out = self.fc(out)
        return out



class frame_based_CASIA_dataset_fusion(Dataset):
    """docstring for data"""
    def __init__(self, txt_path_list, size='256', transform=None):
        imgs_collection = []
        for txt_path in txt_path_list:
            fh = open(txt_path, 'r')
            imgs = []
            for line in fh:
                line = line.rstrip()
                contents = line.split(',')
                imgs.append((contents[0], int(contents[1])))
            imgs_collection.append(imgs)
        self.imgs_collection = imgs_collection
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        imgs_fr = []
        for i in range(len(self.imgs_collection)):
            fn, label = self.imgs_collection[i][index]

            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            imgs_fr.append(img)

        imgs_fr = torch.stack(imgs_fr)

        return imgs_fr,label

    def __len__(self):
        return len(self.imgs_collection[0])


if __name__ == '__main__':
    print('Training with feature-level-fusion resnet18')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        # torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform_train = transforms.Compose([
        # transforms.CenterCrop(512)
        # transforms.RandomResizedCrop(512),
        # transforms.RandomRotation(360),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        # transforms.CenterCrop(512)
        # transforms.RandomResizedCrop(512),
        # transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    chosen_face_regions = args.face_regions[1:-1].split(',')
    print('Chosen face regions:{}'.format(chosen_face_regions))
    num_of_fusion_regions = len(chosen_face_regions)

    train_txt_list = [ '../../train_test_info/numf5/train_' + fr + '_20_1.txt' for fr in chosen_face_regions]
    test_txt_list = [ '../../train_test_info/numf5/test_' + fr + '_30_1.txt' for fr in chosen_face_regions]

    train_data = frame_based_CASIA_dataset_fusion(train_txt_list, 256, transform_train)
    test_data = frame_based_CASIA_dataset_fusion(test_txt_list, 256, transform_test)

    # record file path
    rec_fpath = args.rec_dir + '/' +'fusion_'+ '_'.join(chosen_face_regions) + '_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
    f = open(rec_fpath, "a")
    f.write('ARGS:{}\n'.format(args))
    f.close()

    print('ARGS:{}\n'.format(args))

    # net load
    if use_gpu:
        net = FLF_resnet18(sub_net_num=num_of_fusion_regions).cuda()
    else:
        net = FLF_resnet18(sub_net_num=num_of_fusion_regions)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    dataloader_train = DataLoader(
        train_data, batch_size=args.train_batch, shuffle=True, num_workers=1)
    dataloader_test = DataLoader(
        test_data, batch_size=args.test_batch, shuffle=False, num_workers=1)

    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        total_loss = 0
        total_correct = 0
        total = 0
        # train
        net.train()
        print('number of batch:{}'.format(len(dataloader_train)))
        for id, item in tqdm(enumerate(dataloader_train)):
            data, label = item

            if use_gpu:
                data = data.cuda()
                label = label.cuda()

            # print(data.shape)
            out = net(data)
            _, predicted = torch.max(out, 1)
            loss = criterion(out, label)
            correct = (predicted.cpu().numpy() == label.cpu().numpy()).sum()
            total_correct += correct
            total_loss = total_loss + loss.item()
            total += label.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(label)
            # print(predicted)
            # print(correct)
            # print(total)

            print('id:%d loss:%f correct:%.2f total correct perc:%.3f' % (id, loss.item(),correct/args.train_batch, total_correct / total))
        print("total loss:%f" % (total_loss))
        train_accu = total_correct / total
        train_loss = total_loss

        # test
        with torch.no_grad():
            net.eval()
            total = 0
            total_loss = 0
            correct = 0

            apce = 0
            bpce = 0
            ap_total = 0
            bp_total = 0

            props = []
            labels_rec = []

            for id, item in tqdm(enumerate(dataloader_test)):
                data, label = item

                if use_gpu:
                    data = data.cuda()
                    label = label.cuda()


                out = net(data)

                _, predicted = torch.max(out, 1)
                total += label.size(0)
                correct += (predicted.cpu().numpy() == label.cpu().numpy()).sum()
                loss_test = criterion(out, label)
                total_loss += loss_test.item()

                ap_total += (label.cpu().numpy() == 1).sum()
                bp_total += (label.cpu().numpy() == 0).sum()
                if predicted.cpu().numpy() == 0 and label.cpu().numpy() == 1:
                    apce+=1
                elif predicted.cpu().numpy() == 1 and label.cpu().numpy() == 0:
                    bpce += 1

                # props_ = softmax(out)
                props.append(softmax(out.cpu()))
                labels_rec.append(label.cpu().numpy())

            print('epoch:{}\ttest accuracy:{}\t loss:{}'.format(epoch,correct / total,total_loss))
            print('APCER:{:.4f}\tBPCER:{:.4f}'.format(apce/ap_total,bpce/bp_total))

            props_ = np.array(props)
            resize_props_ = props_.reshape((-1,2))

            f = open(rec_fpath, "a")
            f.write('\nepoch:{}\n'.format(epoch))
            for i in range(resize_props_.shape[0]):
                f.write('{},{},{}\n'.format(resize_props_[i,0],resize_props_[i,1],labels_rec[i]))
            # record the train loss & test loss
            f.write('train accuracy:{:.6ff}\ttrain loss:{:.6f}\n'.format(train_accu,train_loss))
            f.write('test accuracy:{:.6f}\ttest loss:{:.6f}\n'.format(correct / total,total_loss))
            f.write('APCER:{:.4f}\tBPCER:{:.4f}\n'.format(apce/ap_total,bpce/bp_total))
            f.close()


