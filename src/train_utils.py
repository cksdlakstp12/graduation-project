from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pickle
from collections import defaultdict

import config
from model import SSD300, SSD300_3Way

def initialize_state(n_classes, train_conf):
    model = SSD300(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_with_new_optim(train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    
    return model, optimizer, optim_scheduler, start_epoch, None
    
def load_state(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state(args.n_classes, train_conf)
    else:
        if "teacher_weights.pth.tar071" in checkpoint:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_with_new_optim(train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint(train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state(config, student_checkpoint)
    teacher_state = load_state(config, teacher_checkpoint)

    return *student_state, *teacher_state

def initialize_state_3way(n_classes, train_conf):
    model = SSD300_3Way(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5), int(train_conf.epochs * 0.9)],
                                                            gamma=0.1)
    return model, optimizer, optim_scheduler, train_conf.start_epoch, None

def load_state_from_checkpoint_3way(n_classes, train_conf, checkpoint):
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    optim_scheduler = None
    if optimizer is not None:
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(train_conf.epochs * 0.5)], gamma=0.1)
    return model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_3way_with_new_optim_(n_classes, train_conf, checkpoint):
    init_model, optimizer, optim_scheduler, train_conf.start_epoch, _ = initialize_state(n_classes, train_conf)

    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    pret_model = checkpoint['model']
    
    for pret_name, pret_param in pret_model.named_parameters():
        init_param = dict(init_model.named_parameters())[pret_name]
        init_param.data.copy_(pret_param.data)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in init_model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5)],
                                                            gamma=0.1)

    return init_model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_from_checkpoint_3way_with_new_optim(n_classes, train_conf, checkpoint):
    init_model, optimizer, optim_scheduler, train_conf.start_epoch, _ = initialize_state_3way(n_classes, train_conf)

    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
    pret_model = checkpoint['model']
    
    for pret_name, pret_param in pret_model.named_parameters():
        init_param = dict(init_model.named_parameters())[pret_name]
        init_param.data.copy_(pret_param.data)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in init_model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                        {'params': not_biases}],
                                lr=train_conf.lr,
                                momentum=train_conf.momentum,
                                weight_decay=train_conf.weight_decay,
                                nesterov=False)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(train_conf.epochs * 0.5)],
                                                            gamma=0.1)

    return init_model, optimizer, optim_scheduler, start_epoch, train_loss

def load_state_3way(config, checkpoint): 
    args = config.args
    train_conf = config.train

    # Initialize model or load checkpoint
    if checkpoint is None:
        model, optimizer, optim_scheduler, start_epoch, train_loss = initialize_state_3way(args.n_classes, train_conf)
    else:
        if "teacher" in checkpoint:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way_with_new_optim_(args.n_classes, train_conf, checkpoint)
        else:
            model, optimizer, optim_scheduler, start_epoch, train_loss = load_state_from_checkpoint_3way_with_new_optim(args.n_classes, train_conf, checkpoint)

    return (model, optimizer, optim_scheduler, start_epoch, train_loss)

def load_SoftTeacher_3way(config):
    student_checkpoint = config.soft_teacher.student_checkpoint
    teacher_checkpoint = config.soft_teacher.teacher_checkpoint

    # load student and teacher state
    student_state = load_state_3way(config, student_checkpoint)
    teacher_state = load_state_3way(config, teacher_checkpoint)

    return *student_state, *teacher_state

def create_dataloader(config, dataset_class, sample_mode = None, **kwargs):
    if kwargs["condition"] == "train":
        if sample_mode == "two":
            sample = "Labeled"
            dataset = dataset_class(config.args, sample = sample,**kwargs)
            L_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            sample = "Unlabeled"
            dataset = dataset_class(config.args, sample = sample, **kwargs)
            U_loader = DataLoader(dataset, batch_size=int(config.train.batch_size/2), shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
            return dataset, L_loader, U_loader
        else: 
            dataset = dataset_class(config.args, **kwargs)
            loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True,
                                num_workers=config.dataset.workers,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)  # note that we're passing the collate function here
    else:
        dataset = dataset_class(config.args, **kwargs)
        test_batch_size = config.args["test"].eval_batch_size * torch.cuda.device_count()
        loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                              num_workers=config.dataset.workers,
                              collate_fn=dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here
    return dataset, loader

def converter(originpath, changepath, wantname):
    # Loading the 90percents.txt file and creating a dictionary where keys are the index
    with open("./imageSets/" + originpath, 'r') as f:
        data_90 = {idx+1: line.strip() for idx, line in enumerate(f)}

    # Loading the test2.txt file
    with open(changepath, 'r') as f:
        data_test2 = f.readlines()

    # Replacing the first number of each line in test2.txt with corresponding line in 90percents.txt
    data_test2_new = []
    for line in data_test2:
        items = line.split(',')
        index = int(items[0])
        items[0] = data_90[index]
        data_test2_new.append(','.join(items))

    # Writing the new data into a new file
    with open(wantname, 'w') as f:
        for line in data_test2_new:
            f.write(line)

def soft_update(teacher_model, student_model, tau):
    """
    Soft update model parameters.
    θ_teacher = τ*θ_student + (1 - τ)*θ_teacher

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    :param tau: interpolation parameter (0.001 in your case)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(tau*student_param.data + (1.0-tau)*teacher_param.data)

def copy_student_to_teacher(teacher_model, student_model):
    """
    Copy student model to teacher model.
    θ_teacher = θ_student

    :param teacher_model: PyTorch model (Teacher)
    :param student_model: PyTorch model (Student)
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(student_param.data)

class EMAScheduler():
    def __init__(self, config):
        self.use_scheduler = config.ema.use_scheduler
        self.start_tau = config.ema.tau
        self.scheduling_start_epoch = config.ema.scheduling_start_epoch
        self.max_tau = config.ema.max_tau
        self.min_tau = config.ema.min_tau
        self.last_tau = config.ema.tau

    @staticmethod
    def calc_tau(epoch, tau):
        tau = tau
        return tau
    
    def get_tau(self, epoch):
        if not self.use_scheduler:
            return self.start_tau
        else:
            new_tau = EMAScheduler.calc_tau(epoch, self.last_tau)
            return new_tau

def split_features(features, len_L):
    L_features = [feature[:len_L] for feature in features]
    U_features = [feature[len_L:] for feature in features]
    return L_features, U_features

def translate_coordinate(box, feature_w, feature_h):
    x1, y1, x2, y2 = box
    x1 = int(x1 * feature_w)
    y1 = int(y1 * feature_h)
    x2 = int(x2 * feature_w)
    y2 = int(y2 * feature_h)

    if x1 == x2:
        if x2 >= feature_w: x1 -= 1
        else: x2 += 1

    if y1 == y2:
        if y2 >= feature_h: y1 -= 1
        else: y2 += 1

    return x1, y1, x2, y2

def compute_gap_batch(features, box, img_idx):    
    bbox_gaps = []
    for feature in features:
        x1, y1, x2, y2 = translate_coordinate(box, feature.size(3), feature.size(2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:  # 너비나 높이가 0 이하인 경우를 건너뜁니다.
            continue
        cropped_feature = feature[img_idx, :, y1:y2, x1:x2]
        if cropped_feature.size(1) > 0 and cropped_feature.size(2) > 0:
            gap = F.avg_pool2d(cropped_feature, kernel_size=cropped_feature.size()[1:]).view(feature.size(1))
            bbox_gaps.append(gap)

    if bbox_gaps:
        return torch.mean(torch.stack(bbox_gaps, dim=0), dim=0)
    return 

def calc_contrastive_loss(pos, neg, tau=0.1):
    numerator = torch.sum(torch.exp(pos / tau))
    denominator = torch.sum(torch.exp(pos / tau)) + torch.sum(torch.exp(neg / tau))

    contrastive_loss = -torch.log(numerator / denominator)

    return contrastive_loss

def calc_weight_by_GAPVector_distance(features, GT, PL, len_L, input_size):
    ori_h, ori_w = input_size
    L_features, U_features = split_features(features, len_L)

    GTs, PLs = [], []
    per_image_mean_gaps_GT = []
    for idx, boxes in enumerate(GT):
        mean_gaps = [compute_gap_batch(L_features, box, idx) for box in boxes[1:]]
        mean_gaps = [gap for gap in mean_gaps if gap is not None]
        if mean_gaps:  # Check if mean_gaps is not empty
            GTs += mean_gaps
            per_image_mean_gaps_GT.append(torch.mean(torch.stack(mean_gaps), dim=0))

    per_image_mean_gaps_PL = []
    for idx, boxes in enumerate(PL):
        mean_gaps = [compute_gap_batch(U_features, box.numpy(), idx) for box in boxes]
        mean_gaps = [gap for gap in mean_gaps if gap is not None]
        if mean_gaps:  # Check if mean_gaps is not empty
            PLs += mean_gaps
            per_image_mean_gaps_PL.append(torch.mean(torch.stack(mean_gaps), dim=0))

    if per_image_mean_gaps_GT and per_image_mean_gaps_PL:  # Check if both lists are not empty
        GT_vectors = torch.stack(GTs, dim=0)
        contrastive_loss = torch.tensor(0.0, requires_grad=True)
        
        for PL_vector in PLs:
            cos_sim = F.cosine_similarity(PL_vector.unsqueeze(0), GT_vectors, dim=1)
            cos_pos, cos_neg = [], []
            max_cos = torch.max(cos_sim)
            if max_cos > config.train.tau1:
                cos_pos.append(max_cos)  # append max_cos directly as it retains grad
            elif max_cos < config.train.tau2:
                cos_neg.append(max_cos)  # append max_cos directly as it retains grad

        if len(cos_pos) != 0 and len(cos_neg) != 0:
            pos = torch.stack(cos_pos) if cos_pos else torch.tensor([], requires_grad=True)
            neg = torch.stack(cos_neg) if cos_neg else torch.tensor([], requires_grad=True)

            if len(pos) != 0 and len(neg) != 0:
                contrastive_loss = contrastive_loss + calc_contrastive_loss(pos, neg)

        per_image_mean_gaps_GT = torch.mean(torch.stack(per_image_mean_gaps_GT), dim=0)
        per_image_mean_gaps_PL = torch.mean(torch.stack(per_image_mean_gaps_PL), dim=0)    
        
        mse = torch.sqrt(torch.sum((per_image_mean_gaps_GT - per_image_mean_gaps_PL) ** 2, dim=0))
        mse_norm = torch.mean(mse).item()
        weight = np.exp(-mse_norm)
        return weight, contrastive_loss
    else:
        return 0.0, torch.tensor(0.0, requires_grad=True) # Default weight

def calc_unvis_unlwir_weight_by_loss(vis_un_loss, lwir_un_loss):
    with torch.no_grad():
        un_vis_loss_value = vis_un_loss.item()
        un_lwir_loss_value = lwir_un_loss.item()
        un_sum_loss = un_vis_loss_value + un_lwir_loss_value
        if un_sum_loss == 0:
            return 0, 0
        un_vis_weight = un_vis_loss_value / un_sum_loss
        un_lwir_weight = un_lwir_loss_value / un_sum_loss
        return un_vis_weight, un_lwir_weight

def create_relation_matrix(features, len_L, device):
    with torch.no_grad():
        _, U_features = split_features(features, len_L)
        
        relation_matrixes = torch.FloatTensor([])
        for feature in U_features:
            feature_flat = feature.view(len_L, -1)

            relation_matrix = torch.zeros(len_L, len_L)

            for i in range(len_L):
                for j in range(len_L):
                    relation_matrix[i, j] = F.cosine_similarity(feature_flat[i].unsqueeze(0),
                                                                feature_flat[j].unsqueeze(0),
                                                                dim=1)
            relation_matrixes = torch.cat((relation_matrixes, relation_matrix.unsqueeze(dim=0)), dim=0)
        return relation_matrixes.to(device)