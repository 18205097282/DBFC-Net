import torch
import torch.nn as nn
import scipy.spatial
import torch.nn.functional as F
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        """传入的（200,2048）,feature_dim给的2048"""
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        #centers是【200，2048】的

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        distmat=torch.sqrt(distmat)
        
        classes = torch.arange(self.num_classes).long()
 
        if self.use_gpu:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
       
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
 

    # def separate_loss(self):
    #     feature1 = F.normalize(self.centers)  # F.normalize只能处理两维的数据，L2归一化
    #     distance = feature1.mm(feature1.t())  # 计算余弦相似度
    #     mask = 1 - torch.eye(self.num_classes).cuda()
    #     loss=(mask*distance).sum()*0.00005
    #     return loss