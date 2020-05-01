import os
import time
import torch
from torch.autograd import Variable
from util import AverageMeter, Log
from rankingloss import *


def train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion, center_loss, optimizer,
          epoch,
          num_epochs):
    print(len(train_loader), len(train_loader1), len(train_loader2), len(train_loader3))
    """     (图片)1499            （视频）1499          （音频）1500             （文本文档）1000"""
    count = 0
    since = time.time()

    running_loss0 = AverageMeter()
    running_loss1 = AverageMeter()
    running_loss2 = AverageMeter()
    running_loss3 = AverageMeter()
    running_loss4 = AverageMeter()
    running_loss5 = AverageMeter()
    running_loss6 = AverageMeter()
    running_loss7 = AverageMeter()
    running_loss = AverageMeter()

    log = Log()
    model.train()

    image_acc = 0
    text_acc = 0
    video_acc = 0
    audio_acc = 0

    for (i, (input, target)), (j, (input1, target1)), (k, (input2, target2)), (p, (input3, target3)) in zip(
            enumerate(train_loader), enumerate(train_loader1), enumerate(train_loader2), enumerate(train_loader3)):
        """（i,j,k,p）  是  （n,n,n,n） n从0到999结束，故，共只迭代1000次！  有问题！"""
        input_var = Variable(input.cuda())
        input_var1 = Variable(input1.cuda())
        input_var2 = Variable(input2.cuda())
        input_var3 = Variable(input3.cuda())

        targets = torch.cat((target, target1, target2, target3), 0)
        targets = Variable(targets.cuda())

        target_var = Variable(target.cuda())
        target_var1 = Variable(target1.cuda())
        target_var2 = Variable(target2.cuda())
        target_var3 = Variable(target3.cuda())
        outputs, feature = model(input_var, input_var1, input_var2, input_var3)
        size = int(outputs.size(0) / 4)
        img = outputs.narrow(0, 0, size)
        vid = outputs.narrow(0, size, size)
        aud = outputs.narrow(0, 2 * size, size)
        txt = outputs.narrow(0, 3 * size, size)

        _, predict1 = torch.max(img, 1)  # 0是按列找，1是按行找
        _, predict2 = torch.max(vid, 1)  # 0是按列找，1是按行找
        _, predict3 = torch.max(txt, 1)  # 0是按列找，1是按行找
        _, predict4 = torch.max(aud, 1)  # 0是按列找，1是按行找
        image_acc += torch.sum(torch.squeeze(predict1.float() == target_var.float())).item() / float(
            target_var.size()[0])
        video_acc += torch.sum(torch.squeeze(predict2.float() == target_var1.float())).item() / float(
            target_var1.size()[0])
        audio_acc += torch.sum(torch.squeeze(predict4.float() == target_var2.float())).item() / float(
            target_var2.size()[0])
        text_acc += torch.sum(torch.squeeze(predict3.float() == target_var3.float())).item() / float(
            target_var3.size()[0])

        loss0 = criterion(img, target_var)
        loss1 = criterion(vid, target_var1)
        loss2 = criterion(aud, target_var2)
        loss3 = criterion(txt, target_var3)

        loss4 = loss0 + loss1 + loss2 + loss3
        loss5 = center_loss(feature, targets) * 0.001

        if (args.loss_choose == 'r'):
            loss6, _ = ranking_loss(targets, feature, margin=1, margin2=0.5, squared=False)
            loss6 = loss6 * 0.1
        else:
            loss6 = 0.0

        loss = loss4 + loss5 + loss6  # +loss7
        # print(loss)
        batchsize = input_var.size(0)
        running_loss0.update(loss0.item(), batchsize)
        running_loss1.update(loss1.item(), batchsize)
        running_loss2.update(loss2.item(), batchsize)
        running_loss3.update(loss3.item(), batchsize)
        running_loss4.update(loss4.item(), batchsize)
        running_loss5.update(loss5.item(), batchsize)
        # running_loss7.update(loss7.item(), batchsize)

        if (args.loss_choose == 'r'):
            running_loss6.update(loss6.item(), batchsize)
        running_loss.update(loss.item(), batchsize)

        optimizer.zero_grad()
        loss.backward()

        for param in center_loss.parameters():
            param.grad.data *= (1. / 0.001)

        optimizer.step()
        count += 1
        if (i % args.print_freq == 0):

            print('-' * 20)
            print('Epoch [{0}/{1}][{2}/{3}]'.format(epoch, num_epochs, i, len(train_loader)))
            print('Image Loss: {loss.avg:.5f}'.format(loss=running_loss0))
            print('Video Loss: {loss.avg:.5f}'.format(loss=running_loss1))
            print('Audio Loss: {loss.avg:.5f}'.format(loss=running_loss2))
            print('Text Loss: {loss.avg:.5f}'.format(loss=running_loss3))
            print('AllMedia Loss: {loss.avg:.5f}'.format(loss=running_loss4))
            print('Center Loss: {loss.avg:.5f}'.format(loss=running_loss5))
            # print('separate Loss: {loss.avg:.5f}'.format(loss=running_loss7))
            if (args.loss_choose == 'r'):
                print('Ranking Loss: {loss.avg:.5f}'.format(loss=running_loss6))
            print('All Loss: {loss.avg:.5f}'.format(loss=running_loss))
            # log.save_train_info(epoch, i, len(train_loader), running_loss)

    print("训练第%d个epoch:" % epoch)
    print("image:", image_acc / len(train_loader3))
    print("text:", text_acc / len(train_loader3))
    print("video:", video_acc / len(train_loader3))
    print("audio:", audio_acc / len(train_loader3))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), "训练了%d个batch" % count)


