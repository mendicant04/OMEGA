import os
import random
from KKKMeans import kmeans, kmeans_predict

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import lr_schedule
import my_loss
from my_models import get_module, LinearAverage, basenet
import torch
import yaml
from data_loader.get_loader import get_loader
from test import test, class_wise_threshold, visualize_source
import easydict

device = torch.device("cuda")
config_file = 'officehome&domainnet_config.yaml'
conf = yaml.load(open(config_file), Loader=yaml.CLoader)
conf = easydict.EasyDict(conf)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


setup_seed(3407)


def train(steps):
    print('train start!')
    source_path = conf.data.dataset.source_path
    target_path = conf.data.dataset.target_path
    batch_size = conf.data.dataloader.batch_size

    data_transforms = {
        'source': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    source_loader, target_loader, eval_loader, test_folder, pro_loader = get_loader(source_path, target_path,
                                                                                    target_path, data_transforms,
                                                                                    batch_size=batch_size,
                                                                                    return_id=True,
                                                                                    balanced=conf.data.dataloader.class_balance
                                                                                    )

    dataset_eval = eval_loader
    n_share = conf.data.dataset.n_share
    num_class = n_share
    G, C1 = get_module.get_model(conf.model.base_model, num_class=num_class,
                                 temp=conf.model.temp)

    G.to(device)
    C1.to(device)

    num_test_data = test_folder.__len__()

    ## Memory
    lemniscate = LinearAverage.LinearAverage(2048, num_test_data, conf.model.temp, conf.train.momentum).to(device)

    params = []
    for key, value in dict(G.named_parameters()).items():
        # print(key, value)
        if value.requires_grad and "features" in key:
            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
        else:
            if 'bias' in key:
                params += [{'params': [value], 'lr': 1.0,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': 1.0,
                            'weight_decay': conf.train.weight_decay}]

    opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                      weight_decay=0.0005, nesterov=True)
    opt_c1 = optim.SGD(list(C1.parameters()), lr=1.0,
                       momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in opt_c1.param_groups:
        param_lr_f.append(param_group["lr"])

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    best_acc = 0.0

    thresholds = np.array([np.log(n_share) / 2 for _ in range(n_share)])
    # print(thresholds)

    num_target_clusters = np.maximum(int(conf.data.dataset.n_share * conf.train.init_clus), 2)
    # print(num_target_clusters)
    target_centroid = torch.zeros(num_target_clusters, 2048).to(device)

    for step in range(1, steps + 1):
        # print(step)
        G.train()
        C1.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)

        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        lr_schedule.inv_lr_scheduler(param_lr_g, opt_g, step,
                                     init_lr=conf.train.lr,
                                     max_iter=conf.train.min_step)
        lr_schedule.inv_lr_scheduler(param_lr_f, opt_c1, step,
                                     init_lr=conf.train.lr,
                                     max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        label_s = label_s.type(torch.LongTensor)
        img_t = data_t[0]
        index_t = data_t[2]
        img_s, label_s = Variable(img_s.to(device)), \
                         Variable(label_s.to(device))
        img_t = Variable(img_t.to(device))
        index_t = Variable(index_t.to(device))
        if len(img_t) < batch_size:
            break
        if len(img_s) < batch_size:
            break
        opt_g.zero_grad()
        opt_c1.zero_grad()
        ## Weight normalizztion
        C1.weight_norm()
        ## Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)

        loss_s = criterion(out_s, label_s)

        feat_t = G(img_t)
        out_t = C1(feat_t)
        bat_size = out_t.shape[0]


        # loss_cluster begin
        feat_t = F.normalize(feat_t)
        if step % conf.train.kmeans_interval == int(conf.train.kmeans_interval / 2):
            target_centroid = lemniscate.get_target_centroid(num_clusters=int(num_class * conf.train.clus_ratio), device
            =device, way=0)
        out_t_s = F.softmax(out_t, dim=1)
        entr = -torch.sum(out_t_s * torch.log(out_t_s), 1).data.cpu().numpy()
        mean_entr = entr.sum() / batch_size
        pred = out_t_s.data.max(1)[1].cpu().numpy()
        class_threshold = thresholds[pred]

        logits = out_t_s.data.max(1)[0].cpu().numpy()
        pred_unk = np.where(entr > class_threshold)[0]

        feat_unk = feat_t[pred_unk]
        unk_idx = kmeans_predict(feat_unk, target_centroid, 'euclidean', device=device)
        unk_idx += num_class
        pred[pred_unk] = unk_idx
        logits[pred_unk] = entr[pred_unk]
        tmp_logit = torch.from_numpy(logits).unsqueeze(dim=1)
        tmp_logit_m = tmp_logit.expand(bat_size, bat_size)
        tmp_logit_T = tmp_logit.T
        tmp_logit_T_m = tmp_logit_T.expand(bat_size, bat_size)
        coef_matrix = torch.min(tmp_logit_T_m, tmp_logit_m)

        tmp_pred = torch.from_numpy(pred).unsqueeze(dim=1)
        tmp_pred_m = tmp_pred.expand(bat_size, bat_size)
        tmp_pred_T = tmp_pred.T
        tmp_pred_T_m = tmp_pred_T.expand(bat_size, bat_size)
        mask_matrix = torch.eq(tmp_pred_T_m, tmp_pred_m).long()
        mask_matrix.fill_diagonal_(0)
        mask_matrix = (mask_matrix * coef_matrix).to(device)

        out_t_s1 = out_t_s.clone()
        out_t_s2 = out_t_s.clone()
        out_t_s1l = out_t_s1.log()
        out_t_s2l = out_t_s2.log()
        batch_kl1 = (out_t_s2 * out_t_s2.log()).sum(dim=1) - torch.einsum('ik, jk -> ij', out_t_s1l, out_t_s2)
        batch_kl2 = (out_t_s1 * out_t_s1.log()).sum(dim=1) - torch.einsum('ik, jk -> ij', out_t_s2l, out_t_s1)
        batch_kl = (batch_kl1 + batch_kl2) / 2
        loss_clus = torch.sum(mask_matrix * batch_kl)

        loss_clus /= len(img_t) * len(img_t)
        # loss_cluster end

        ### Calculate mini-batch x memory similarity
        feat_mat = lemniscate(feat_t)
        ### We do not use memory features present in mini-batch
        feat_mat[:, index_t] = -1 / conf.model.temp
        ### Calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / conf.model.temp
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).bool().to(device)
        feat_mat2.masked_fill_(mask, -1 / conf.model.temp)
        loss_nc = conf.train.eta1 * my_loss.entropy(torch.cat([out_t, feat_mat,
                                                               feat_mat2], 1))
        loss_ent = conf.train.eta1 * my_loss.entropy_margin(out_t, np.log(num_class) / 2,
                                                            conf.train.margin)

        if 'visda' in config_file:
            warm_up = conf.train.warm_up
        else:
            warm_up = conf.train.warm_up * np.log(num_class)
        if mean_entr > warm_up:
            eta2 = conf.train.warm_dec * conf.train.eta2
        else:
            eta2 = conf.train.eta2

        all = loss_s + loss_nc + loss_ent + eta2 * loss_clus
        all.backward()

        opt_g.step()
        opt_c1.step()
        opt_g.zero_grad()
        opt_c1.zero_grad()

        lemniscate.update_weight(feat_t, pred, index_t)

        if step % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '
                  'Loss NC: {:.6f} Loss ENS: {:.6f} Loss_Cluster: {:.6f} Best HOS:{:.3f} \t'.format(
                step, conf.train.min_step,
                100 * float(step / conf.train.min_step),
                loss_s.item(), loss_nc.item(), loss_ent.item(), loss_clus, best_acc))

        if step > 0 and step % conf.test.test_interval == 0:
            acc, per_cls_acc_known, acc_unknown, pseudo_entr = test(step, dataset_eval, n_share, num_class, G, C1,
                                                                    thresholds)
            hos_score = 2.0 * (per_cls_acc_known * acc_unknown) / (per_cls_acc_known + acc_unknown)
            if hos_score > best_acc:
                best_acc = hos_score
                print("Acc:%.4f     per_class_acc_known:%.4f        acc_unknown:%.4f     hos:%.4f " % (
                acc, per_cls_acc_known, acc_unknown, best_acc))

            G.train()
            C1.train()
            thresholds = class_wise_threshold(entropies=pseudo_entr, num_class=n_share, r=conf.train.mov)
    return G, C1


train(conf.train.min_step)
