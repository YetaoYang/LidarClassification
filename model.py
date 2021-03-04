import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool,DataParallel
from torch_cluster import knn
from torch_geometric.nn import knn_interpolate



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)


        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class SAModuleMsg(torch.nn.Module):
    def __init__(self, ratio, rlist, nsamplelist, channelslist):
        super(SAModuleMsg, self).__init__()
        self.ratio = ratio
        self.rlist = rlist
        self.nsamplelist = nsamplelist
        self.convlist = nn.ModuleList()
        for i in range(len(channelslist)):
            self.convlist.append(PointConv(MLP_ResBlock(channelslist[i])))

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        x_new = []
        for i, dil in enumerate(self.rlist):

            N, K = idx.size(-1), self.nsamplelist[i]
            row, col = knn(pos, pos[idx], K * dil, batch, batch[idx]
                              )
            if dil > 1:
                index = torch.randint(K * dil, (N, K), dtype=torch.long,
                                      device=row.device)
                arange = torch.arange(N, dtype=torch.long, device=row.device)
                arange = arange * (K * dil)
                index = (index + arange.view(-1, 1)).view(-1)
                row, col = row[index], col[index]
            edge_index = torch.stack([col, row], dim=0)
            conv = self.convlist[i]
            xi = conv(x, (pos, pos[idx]), edge_index)
            x_new.append(xi)
        pos, batch = pos[idx], batch[idx]
        x_new_contact = torch.cat(x_new,dim=1)
        return x_new_contact, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, channels):
        super(GlobalSAModule, self).__init__()
        self.nn = MLP_ResBlock(channels)

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class MLP_ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(MLP_ResBlock, self).__init__()
        self.channels = channels
        self.linearlist = nn.ModuleList()
        for i in range(1, len(self.channels)-1):
          self.linearlist.append(Seq(Lin(self.channels[i - 1], self.channels[i]), BN(self.channels[i]), ReLU()))
        self.linearlist.append(Seq(Lin(self.channels[-2], self.channels[-1]), BN(self.channels[-1])))
        self.downsample = Seq(Lin(self.channels[0], self.channels[-1]), BN(self.channels[-1]))
        self.relu = ReLU(inplace=True)
    def forward(self, x):
        identity = x
        for i in range (len(self.linearlist)):
           seqconv = self.linearlist[i]
           x = seqconv(x)
        x += self.downsample(identity)
        x = self.relu(x)
        return x

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])



class FPModule(torch.nn.Module):
    def __init__(self, k, channels):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = MLP_ResBlock(channels)

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes, loop_times):
        super(Net, self).__init__()

        self.sa1_module = SAModuleMsg(0.25, [2, 2], [8, 24], [[8 + 3, 16, 32], [8 + 3, 16, 32]])
        self.sa2_module = SAModuleMsg(0.25, [2, 2], [8, 24], [[64 + 3, 128, 128], [64 + 3, 128, 128]])
        self.sa3_module = GlobalSAModule([256 + 3, 512, 512])

        self.fp3_module = FPModule(1, [512 + 256, 512, 256])
        self.fp2_module = FPModule(3, [256 + 64, 256, 128])
        self.fp1_module = FPModule(3, [128 + 8, 128, 128])

        self.crsa1_module = SAModuleMsg(0.25, [2, 2], [8, 24], [[128 + 3, 64, 64], [128 + 3, 64, 64]])
        self.crsa2_module = SAModuleMsg(0.25, [2, 2], [8, 24], [[128 + 3, 128, 128], [128 + 3,  128, 128]])
        self.crsa3_module = GlobalSAModule([256 + 3, 512, 512])

        self.crfp3_module = FPModule(1, [512 + 256, 512, 256])
        self.crfp2_module = FPModule(3, [256 + 128, 256, 128])
        self.crfp1_module = FPModule(3, [128 + 128, 128, 128])

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)
        self.loop = loop_times

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp2_out = self.fp3_module(*sa3_out, *sa2_out)
        fp1_out = self.fp2_module(*fp2_out, *sa1_out)
        fp0_out = self.fp1_module(*fp1_out, *sa0_out)

        crsa0_out = fp0_out
        for i in range(self.loop):

            crsa1_out = self.crsa1_module(*crsa0_out)
            crsa2_out = self.crsa2_module(*crsa1_out)
            crsa3_out = self.crsa3_module(*crsa2_out)

            crfp2_out = self.crfp3_module(*crsa3_out, *crsa2_out)
            crfp1_out = self.crfp2_module(*crfp2_out, *crsa1_out)
            crfp0_out= self.crfp1_module(*crfp1_out, *crsa0_out)
            crsa0_out = crfp0_out
        x,_,_ = crsa0_out
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
