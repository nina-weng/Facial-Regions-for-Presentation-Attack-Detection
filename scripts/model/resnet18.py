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




parser = argparse.ArgumentParser(description='train resnet18 model with frame-based CASIA PAD')
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--size', default='256', choices=['154','140','256','192','combine'])
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--num-epochs', type=int, default=50, help="number of epochs")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
args = parser.parse_args()


class frame_based_CASIA_dataset(Dataset):
    """docstring for data"""
    def __init__(self, txt_path, size='256', transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            contents = line.split(',')
            imgs.append((contents[0], int(contents[1])))
        self.imgs = imgs
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # img = img.unsqueeze(0)

        return img,label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     print("Currently using GPU {}".format(args.gpu_devices))
    #     cudnn.benchmark = True
    #     #torch.cuda.manual_seed_all(args.seed)
    # else:
    #     print("Currently using CPU (GPU is highly recommended)")

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
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(512),
            #transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = frame_based_CASIA_dataset('..\..\\train_test_info\\train_15_r1.txt', 256, transform_train)
    test_data = frame_based_CASIA_dataset('..\..\\train_test_info\\test_5_r1.txt', 256, transform_test)

    # load pre-trained resnet18 model
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512,2,bias=True)

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
            out = net(data)
            _, predicted = torch.max(out, 1)
            loss = criterion(out, label)
            correct = (predicted.numpy() == label.numpy()).sum()
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

        # test
        with torch.no_grad():
            net.eval()
            total = 0
            total_loss = 0
            correct = 0
            for id, item in tqdm(enumerate(dataloader_test)):
                data, label = item
                out = net(data)

                _, predicted = torch.max(out, 1)
                total += label.size(0)
                correct += (predicted.numpy() == label.numpy()).sum()
                loss_test = criterion(out, label)
                total_loss += loss_test.item()
            print('test accuracy:{}\t loss:{}'.format(correct / total,total_loss))


