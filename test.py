import torch
import torch.nn.functional as F
import numpy as np


def class_wise_threshold(entropies, num_class, r=0.1):

    min_threshold = np.log(num_class) * (0.5 - r)
    max_threshold = np.log(num_class) * (0.5 + r)
    min_entropy = np.min(entropies)
    max_entropy = np.max(entropies)

    threshold = []

    for i in range(len(entropies)):
        ratio = (entropies[i] - min_entropy) / (max_entropy - min_entropy)
        threshold.append(min_threshold + ratio * (max_threshold - min_threshold))

    threshold = np.array(threshold)

    return threshold



def test(step, dataset_test, n_share, unk_class, G, C1, threshold):
    G.eval()
    C1.eval()
    correct = 0
    size = 0
    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    raw_pred = []
    all_gt = []
    all_entr = []
    thresholds = threshold

    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()

            feat = G(img_t)
            out_t = C1(feat)
            out_t = F.softmax(out_t, dim=1)
            pred = out_t.data.max(1)[1].data.cpu().numpy()
            raw_pred += list(pred)
            class_threshold = thresholds[pred]
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            all_entr += list(entr)

            pred_unk = np.where(entr > class_threshold)

            k = label_t.data.size()[0]

            pred[pred_unk[0]] = unk_class
            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
            size += k
    per_class_acc = per_class_correct / per_class_num
    per_class_acc_known = float(per_class_acc[:-1].mean())
    acc_unknown = float(per_class_acc[-1])

    all_entr = np.array(all_entr)
    raw_pred = np.array(raw_pred)
    mean_entropy = np.zeros(n_share).astype(np.float32)
    for i in range(n_share):
        idx = np.where(raw_pred == i)[0]
        ents = all_entr[idx]
        if ents.shape[0] == 0:
            mean_entropy[i] = np.log(n_share) / 2
        else:
            mean_entropy[i] = ents.mean()
    if step % 10 == 0:
        # pass
        print("Acc:%.4f     per_class_acc_known:%.4f        acc_unknown:%.4f" % (100.0 * correct / size, per_class_acc_known, acc_unknown))
    return 100.0 * correct / size, per_class_acc_known, acc_unknown, mean_entropy

def visualize(dataset_test, G, C1):
    G.eval()
    C1.eval()
    output_feat = None
    output_label = None
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_test):
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            if output_feat is None:
                output_feat = feat.detach().cpu().numpy().squeeze()
                output_label = label_t.detach().cpu().numpy().squeeze()
            else:
                output_feat = np.concatenate((output_feat, feat.detach().cpu().numpy().squeeze()))
                output_label = np.concatenate((output_label, label_t.detach().cpu().numpy().squeeze()))
    with open('feat_omega.npy', 'wb') as f:
        np.save(f, output_feat)

    np.savetxt('plabel_omega.txt', output_label)

def visualize_source(source_loader, G, C1):
    G.eval()
    C1.eval()
    output_feat = None
    output_label = None
    with torch.no_grad():
        for batch_idx, data in enumerate(source_loader):
            img_t, label_t = data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            if output_feat is None:
                output_feat = feat.detach().cpu().numpy().squeeze()
                output_label = label_t.detach().cpu().numpy().squeeze()
            else:
                output_feat = np.concatenate((output_feat, feat.detach().cpu().numpy().squeeze()))
                output_label = np.concatenate((output_label, label_t.detach().cpu().numpy().squeeze()))
    with open('source_feat_omega.npy', 'wb') as f:
        np.save(f, output_feat)

    np.savetxt('source_label_omega.txt', output_label)