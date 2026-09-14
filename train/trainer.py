import numpy as np
import torch
import cv2, os
import random
from PIL import Image
from einops import rearrange

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import logging

from data.dataset_loader import get_dataset
from config.config_loader.config_base import Config_base
from model.net import Net_loader
from tools.distributed import Distributed_utils as du
from train.loss_loader import get_loss
from tools.checkpoint import Checkpoint_io
from train.optimizer_loader import get_optimizer, get_lr_scheduler
from tools.param_constant import *
from train.dataloader_wrapper import DataLoader_wrapper
from tqdm import tqdm
from train.gradient_alignment import GradientAlignment


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.GPUS = cfg.train.gpus
        if self.GPUS == 'cpu':
            self.NUM_GPUS = 0
        elif isinstance(self.GPUS, list):
            self.NUM_GPUS = len(self.GPUS)
            self.GPU_BASE = self.GPUS[0]
        else:
            self.NUM_GPUS = 1
            self.GPU_BASE = self.GPUS
        self.BATCH_SIZE_PER_GPU = cfg.train.batch_per_gpu

        self.SEED = cfg.train.seed

        self.NUM_EPOCHS = cfg.train.num_epochs
        self.RESET_EPOCH = cfg.train.get(RESET_EPOCH, False) # Zheng(23-08-03): set default value to False for continue training from existing epoch
        self.LOG_PERIOD = cfg.train.log_period
        self.VAL_PERIOD = cfg.train.get(VAL_PERIOD, 1)
        self.NO_VAL_BEFORE = cfg.train.get(NO_VAL_BEFORE, 0)
        self.INFER_TEST = cfg.train.get(INFER_TEST, False)
        self.ITERS_LIMIT = cfg.train.get(ITERS_LIMIT, 0)  # 0 means no limit (normal training)

        self.CKPT_DIR = cfg.train.get(CKPT_DIR, None)
        self.CKPT_PERIOD = cfg.train.get(CKPT_PERIOD, 0) # 0 means to preserve the ckpt with minimal val loss ONLY; -1 means never saving ckpts
        self.CKPT_AUTO_CLEAR = cfg.train.get(CKPT_AUTO_CLEAR, False)

        self.VAL_PRINT_CAR = cfg.train.get(VAL_PRINT_CAR, False)

        # Init actions
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        # Parallel
        du.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.local_rank = du.get_rank()
        self.device = self.get_device()

        # Construct the Network & Load ckpt
        self.net_loader = Net_loader(cfg.net)
        self.start_epoch = 0 if self.RESET_EPOCH else self.net_loader.start_epoch
        self.model = self.net_loader.model
        self.parallel = self.NUM_GPUS>1
        if self.parallel:
            # Parallelize BN layers
            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device) # might causes hang since if some gpu is out-of-sync, no idea why, using normal BN layer now
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else:
            self.model = self.model.to(self.device)
        self.n_input = len(cfg.net.input)

        # Dataset instances
        self.dataset_train = get_dataset(cfg.data.dataset_train)  # Dataset object
        self.EXPLICIT_SHUFFLE = hasattr(self.dataset_train, 'sampler') or hasattr(self.dataset_train, 'hybrid')
        self.num_views_train = cfg.train.get(NUM_VIEWS, 1)
        self.flip_every_other_view = cfg.train.get(FLIP_EVERY_OTHER_VIEW, False)
        self.glasses_latter_half_views = cfg.train.get(GLASSES_LATTER_HALF_VIEWS, False)
        self.share_t2d = cfg.train.get(SHARE_T2D, False)
        self.dataloader_train = DataLoader_wrapper(self.dataset_train, self.BATCH_SIZE_PER_GPU,
            num_views=self.num_views_train,
            flip_every_other_view=self.flip_every_other_view,
            glasses_latter_half_views=self.glasses_latter_half_views,
            parallel=self.parallel,
            share_t2d=self.share_t2d)
        self.n_iter_train = len(self.dataloader_train)
        self.logger.info("Loaded Dataset_train from: %s" % cfg.data.dataset_train.type)
        self.logger.info("len(D_train) = {}".format(len(self.dataset_train)))
        
        if getattr(cfg.data, "dataset_val", None) is not None:
            self.dataset_val = get_dataset(cfg.data.dataset_val, dataset_base=self.dataset_train)  # Dataset object; use `dataset_base` to avoid loading the GT npy again
            self.num_views_val = cfg.data.dataset_val.args.get(NUM_VIEWS, 1)
            self.dataloader_val = DataLoader_wrapper(self.dataset_val, self.BATCH_SIZE_PER_GPU, split='val', num_views=self.num_views_val, parallel=self.parallel)
            self.n_iter_val = len(self.dataloader_val)
            self.logger.info("Loaded Dataset_val from: %s" % cfg.data.dataset_val.type)
            self.logger.info("len(D_val) = {}".format(len(self.dataset_val)))
        else:
            self.logger.info("No dataset_val section in cfg. Will skip validation!")
            self.dataset_val = None
    
        # Construct optimizers
        self.optimizers, self.lr_schedulers = self.initialize_optimizers(self.net_loader, self.cfg.train.optimizer)
        self.logger.info("Optimizers & LR_Schedulers initialized.")
        
        # Construct variable losses
        self.loss_cfg, self.loss_modules, self.loss_fns = self.initialize_loss()
        self.print_val_loss_detail = cfg.train.get(PRINT_VAL_LOSS_DETAIL, False)
        self.print_train_loss_detail = cfg.train.get(PRINT_TRAIN_LOSS_DETAIL, False)

        # CKPT
        self.cio = Checkpoint_io(self.CKPT_DIR, parallel=self.NUM_GPUS>1) if du.is_master_proc() and self.CKPT_DIR else None
        if self.cio and self.CKPT_AUTO_CLEAR:
            self.cio.clear_dir()
        self.logger.info("Ckpt-IO initialized.")

        # Tensorboard
        self.tb = SummaryWriter(self.CKPT_DIR) if du.is_master_proc() and self.CKPT_DIR else None
        self.logger.info("TB initialized.")

        # Mehdi's modules
        if self.cfg.train.get('grad_alignment', False):                
            print('Gradient Alignment will be used in training process')
            print('Aligned Losses:', self.cfg.train.list_aligned_losses)
            print('Main Loss:', self.cfg.train.main_loss)
            self.ga = GradientAlignment(self.cfg.train.list_aligned_losses, self.cfg.train.main_loss, self.model)

        if self.cfg.train.get('stop_grad_scl', False):
            print('Stop Gradient will be used for SSL')

        self.logger.info("[Trainer]Init done.")

    def get_device(self):
        if self.NUM_GPUS==0:
            self.logger.info("[Trainer]Device = CPU")
            return torch.device('cpu')
        elif self.NUM_GPUS==1:
            self.logger.info("[Trainer]Device = SINGLE GPU: {}".format(self.GPUS))
            return torch.device(f'cuda:{self.GPU_BASE}')
        else:
            self.logger.info("[Trainer]Device = MULTI-GPU: {}".format(self.GPUS))
            return torch.cuda.current_device()

    def train(self):
        min_loss_val = 1e6

        for epoch in range(self.start_epoch, self.NUM_EPOCHS):
            self.logger.info("=======EPOCH #%d  =======" % epoch)

            # epoch starter - shuffle IDs and refresh the DataLoader(s)
            # if self.EXPLICIT_SHUFFLE:
            self.dataset_train.shuffle()
            if self.num_views_train > 1:
                self.dataloader_train.synchronize()
            self.dataloader_train.init_dataloader()

            loss_train, losses_dict_train = self.train_epoch()
            self.logger.info("Train Loss @EPOCH {} = {:.4f}".format(epoch, loss_train))
            if self.tb:
                self.tb.add_scalar("Loss_train", loss_train, global_step=epoch)
                if len(losses_dict_train) > 1:
                    for key, value in losses_dict_train.items():
                        self.tb.add_scalar("Loss_train_{}".format(key), value, global_step=epoch)

                for key, lrs in self.lr_schedulers.items():
                    self.tb.add_scalar("LR_{}".format(key), lrs.get_last_lr()[0], global_step=epoch)

            if self.cio and self.CKPT_PERIOD > 0 and epoch % self.CKPT_PERIOD == self.CKPT_PERIOD - 1:
                self.cio.save_ckpt(self.model, epoch, cfg=None)
                self.logger.info("CKPT saved @EPOCH {}".format(epoch))

            if self.dataset_val is not None and self.VAL_PERIOD > 0 and epoch >= self.NO_VAL_BEFORE and epoch % self.VAL_PERIOD == self.VAL_PERIOD - 1:
                loss_val, losses_dict_val = self.val_epoch(epoch=epoch, print_car=self.VAL_PRINT_CAR)
                self.logger.info("Val Loss @EPOCH {} = {:.4f}".format(epoch, loss_val))
                if self.tb:
                    self.tb.add_scalar("Loss_val", loss_val, global_step=epoch)
                    if len(losses_dict_val) > 1:
                        for key, value in losses_dict_val.items():
                            self.tb.add_scalar("Loss_val_{}".format(key), value, global_step=epoch)
                if loss_val < min_loss_val:
                    min_loss_val = loss_val
                    if self.cio and self.CKPT_PERIOD == 0:
                        self.cio.save_ckpt(self.model, epoch, cfg=None)
                        self.logger.info("[train]CKPT saved @EPOCH {}".format(epoch))

            if self.INFER_TEST:
                self.infer_test()

            # update LR
            for lrs in self.lr_schedulers.values():
                lrs.step()
        
    def train_epoch(self):

        self.model.train()
        
        losses = []
        losses_dict = {}
        for iter, (X, Y_gt) in tqdm(enumerate(self.dataloader_train)):
            if self.ITERS_LIMIT > 0 and iter >= self.ITERS_LIMIT:
                break

            X = [x.float() for x in X]
            if self.NUM_GPUS!=0:
                X = [x.to(self.device) for x in X]

            loss, loss_cpu, losses_dict_iter = self.forward_loss(self.model(X), Y_gt)
            self.backward(loss, losses_dict_iter)
            losses_dict_iter = {k:v.cpu().detach().numpy() for k, v in losses_dict_iter.items()}


            if iter % self.LOG_PERIOD == 0:
                self.logger.info("Train Loss @Iter {}/{} = {:.4f}".format(iter, self.n_iter_train, loss_cpu))

            losses.append(loss_cpu)
            for key, value in losses_dict_iter.items():
                if key not in losses_dict:
                    losses_dict[key] = []
                losses_dict[key].append(value)

        for key, value in losses_dict.items():
            losses_dict[key] = np.mean(np.array(value))

        return np.mean(np.array(losses)), losses_dict

    @torch.no_grad()
    def val_epoch(self, epoch, print_car=False):

        self.model.eval()

        losses = []
        losses_dict = {}
        for iter, (X, Y_gt) in tqdm(enumerate(self.dataloader_val)):
            if self.ITERS_LIMIT > 0 and iter >= self.ITERS_LIMIT:
                break
            
            X = [x.float() for x in X]
            if self.NUM_GPUS!=0:
                X = [x.to(self.device) for x in X]

            model_output = self.model(X)

            # Just for debugging ckpts with CAR loss - print the gaze head
            if print_car and iter==0:
                print("debug gaze output")
                gaze_output = model_output['head'][0].cpu().detach().numpy().squeeze()
                for j in range(8):
                    print("PITCH")
                    print(gaze_output[j, :9]) # 9-d
                    print("YAW")
                    print(gaze_output[j, 9:]) # 13-d

            if 'head_seg' in model_output.keys():
                self.visualize_gazemap(model_output, X, epoch, iter)

            _, loss_cpu, losses_dict_iter = self.forward_loss(model_output, Y_gt, train=False)
            losses_dict_iter = {k:v.cpu().detach().numpy() for k, v in losses_dict_iter.items()}

            if iter % self.LOG_PERIOD == 0:
                self.logger.info("Val Loss @Iter {}/{} = {:.4f}".format(iter, self.n_iter_val, loss_cpu))

            losses.append(loss_cpu)
            for key, value in losses_dict_iter.items():
                if key not in losses_dict:
                    losses_dict[key] = []
                losses_dict[key].append(value)

        for key, value in losses_dict.items():
            losses_dict[key] = np.mean(np.array(value))

        return np.mean(np.array(losses)), losses_dict
    
    @torch.no_grad()
    def visualize_gazemap(self, model_output, input_x, epoch, iter):
        gazemap_dir = os.path.join(self.CKPT_DIR,  'gazemap', str(epoch))
        os.makedirs(gazemap_dir, exist_ok=True)

        seg_out = model_output['head_seg'][0] # [4, 8, 224, 224]

        eye_L = torch.nn.functional.softmax(seg_out[0, :2, :, :], dim=0).cpu().detach().numpy()
        iris_L = torch.nn.functional.softmax(seg_out[0, 2:4, :, :], dim=0).cpu().detach().numpy()
        eye_R = torch.nn.functional.softmax(seg_out[0, 4:6, :, :], dim=0).cpu().detach().numpy()
        iris_R = torch.nn.functional.softmax(seg_out[0, 6:, :, :], dim=0).cpu().detach().numpy()

        eye_L = rearrange(eye_L, 'd h w -> h w d')
        iris_L = rearrange(iris_L, 'd h w -> h w d')
        eye_R = rearrange(eye_R, 'd h w -> h w d')
        iris_R = rearrange(iris_R, 'd h w -> h w d')

        eye = Image.fromarray(((eye_L[:, :, 1] + eye_R[:, :, 1])*255).astype('uint8')).convert('L')
        iris = Image.fromarray(((iris_L[:, :, 1] + iris_R[:, :, 1])*255).astype('uint8')).convert('L')

        eye.save(os.path.join(gazemap_dir, f"{iter}_eyemap.png"))
        iris.save(os.path.join(gazemap_dir, f"{iter}_irismap.png"))
        
        input_image = (input_x[0][0, ...]).cpu().detach().numpy()
        input_image = rearrange(input_image, 'd h w -> h w d')
        input_image = ((input_image + 1.0)*127.5).astype('uint8')
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(gazemap_dir, f"{iter}_input_image.png"), input_image)


    @torch.no_grad()
    def infer_test(self, n_iters=10):
        self.model.eval()

        self.logger.info("Inference test")
        torch.manual_seed(0)
        for iter, (X, _) in enumerate(self.dataloader_val):
            if iter >= n_iters:
                break
            self.logger.info("[train]Batch #%d" % iter)
            Y_pred = self.model([X]).cpu().detach().numpy()
            for i in range(self.BATCH_SIZE_PER_GPU):
                self.logger.info("[train]Output #{} = {}".format(i, Y_pred[i]))
        torch.manual_seed(self.SEED)

    @torch.no_grad()
    def make_optimizer(self, cfg, module_params):
        # simple helper that create an optimizer based on passed cfg
        type = cfg.type  
        lr = float(cfg.lr)
        args = cfg.args.copy() if cfg.args is not None else {}
        optimizer = get_optimizer(type)
        # ignore LR_SCHEDULER
        args.pop(LRS, None)
        return optimizer(module_params, lr=lr, weight_decay=1e-6, **args)
    
    @torch.no_grad()
    def make_lrs(self, cfg, optimizer): #lr_scheduler
        # simple helper that create a lr_scheduler based on passed cfg
        type = cfg.type
        if type is None or type == "None": return None
        args = cfg.get(ARGS, {})
        lrs = get_lr_scheduler(type)
        return lrs(optimizer, **args)

    @torch.no_grad()
    def initialize_optimizers(self, net, cfg):
        # interesting enough, to control who you want to optimize, you need physically go into 
        # each module and make sure that you only mess with module and that module specifically
        # so to support individual optimization, we need to loop through the entire list of modules, 
        # and sort them into different optimizers
        
        default = cfg.default

        has_default_lrs = LRS in cfg.default.__dict__.keys()
        if has_default_lrs:
            cfg_default_lrs = cfg.default.__dict__[LRS]

        modules = net.get_all_modules()
        default_list = []
        optimizers = []
        schedulers = {}
        for module_name, module_content in modules.items():
            if module_name in cfg.__dict__:
                module_cfg = cfg.__dict__[module_name]
                optimizer = self.make_optimizer(module_cfg, module_content.parameters())
                optimizers.append(optimizer)
                if LRS in module_cfg.__dict__.keys():
                    lrs = self.make_lrs(module_cfg.__dict__[LRS], optimizer)
                    if lrs is not None: schedulers[module_name] = lrs
                elif has_default_lrs:
                    lrs = self.make_lrs(cfg_default_lrs, optimizer)
                    if lrs is not None: schedulers[module_name] = lrs
            else: # it goes into default_list
                default_list += module_content.parameters()
        optimizer_default = self.make_optimizer(default, default_list)
        optimizers.append(optimizer_default)
        if has_default_lrs:
            lrs = self.make_lrs(cfg_default_lrs, optimizer_default)
            if lrs is not None: schedulers[DEFAULT] = lrs
        return optimizers, schedulers

    def backward(self, total_loss, losses_dict):
        '''
        *** the backward function for SCL always assumes loss2 is SCL loss
        *** the backward function should be developed if both positive and negative SCL loss wants to be used for person independent and dependent modules 
        '''
        
        # in this step, we call the optimization step of every single optimizers we have created + back propagation
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        dict_loss_grads = dict()
        # generate gradients for each loss
        if self.cfg.train.get('grad_alignment', False):
            for name in (self.cfg.train.list_aligned_losses + [self.cfg.train.main_loss]):
                loss_form = self.loss_cfg.loss_train
                loss_name = torch.mul(losses_dict[name], loss_form[name])
                grad_loss = torch.autograd.grad(loss_name, self.model.parameters(), grad_outputs=None, retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)
                dict_loss_grads[name] = list(grad_loss)
                self.model.zero_grad()

            dict_aligned_losses = self.ga.get_aligned_grads(dict_loss_grads)
            # replaces the aux losses with the aligned aux losses in the dictionary of gradients
            for loss_name in dict_loss_grads.keys():
                if loss_name in dict_aligned_losses.keys():
                    dict_loss_grads[loss_name] = dict_aligned_losses[loss_name]
            
        if self.cfg.train.get('stop_grad_scl', False):
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if (dict_loss_grads[self.cfg.train.scl_loss][i] is not None) and ('head_ssl.person_indep_module' not in name):
                    dict_loss_grads[self.cfg.train.scl_loss][i] = 0 * dict_loss_grads[self.cfg.train.scl_loss][i]


        # gather all results for parameters
        for i, (name, param) in enumerate(self.model.named_parameters()):
            for loss_name, val_grad in dict_loss_grads.items():
                if dict_loss_grads[loss_name][i] is not None:
                    if param.grad is None:
                        param.grad = dict_loss_grads[loss_name][i].clone()
                    else:
                        param.grad = torch.add(param.grad, dict_loss_grads[loss_name][i].clone())

        # Zheng(23-01-04): backward() function was missing
        total_loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
    
    @torch.no_grad()
    def initialize_loss(self):
        ## we first construct all loss mentioned, and used in this net config
        loss_cfg = self.cfg.train.loss
        loss_modules = [(name, config) for name, config in loss_cfg.__dict__.items() if isinstance(config, Config_base)]
        loss_fns = {}
        for name, config in loss_modules:
            loss_type = config.type
            loss_args = config.get("args", {})
            loss_fns[name] = get_loss(loss_type, **loss_args)
        return loss_cfg, loss_modules, loss_fns

    def forward_loss(self, Ys_pred, Ys_gt, train=True):
        # We first try to calculate all the loss used in loss composistion,
        # then store the result under losses_dict
        # Nov 11 '22 update: add 3rd output - losses_dict to view each loss term
        losses_dict = {}
        
        # loss_form stores the list of loss terms with cfg.
        # Some loss terms not included in loss_train/val can be skipped
        loss_form = None
        if train:
            loss_form = self.loss_cfg.loss_train
        else: 
            loss_form = self.loss_cfg.loss_val

        # Calculate each loss term
        for name, config in self.loss_modules:
            if name not in loss_form:
                continue
            loss_func = self.loss_fns[name]
            loss_input = []
            # in this step we are preparing the all the input that will be feed into the loss module
            # note that due to the nature of Pytroch, it better to cast everything to a float
            for source, split in config.module_input.items():
                if source == DATASET: 
                    if isinstance(split, list): ## 'dataset: [1, 3, 5]' fetch segmentation map, given indices
                        gt_list = [Ys_gt[i] for i in split]
                        if self.NUM_GPUS!=0:
                            gt_list = [t.cuda(self.device) for t in gt_list]
                        loss_input.append(gt_list)
                    elif len(str(split)) > 1 and int(split) > 12: # "dataset: 16" to fetch segmentation map
                        # segmentation map need to be in the same order as Amirul did, "xx" can't start with 0...
                        # so in config, dataset section, put "id" to take the first position
                        # NOTE: not recommended to use, and won't support 0x / 10 / 11 / 12
                        gt_list = [_label for _label in Ys_gt[int(str(split)[0]):int(str(split)[-1])]]
                        if self.NUM_GPUS!=0:
                            gt_list = [t.cuda(self.device) for t in gt_list]
                        loss_input.append(gt_list)
                    else:
                        gt_list = Ys_gt[split].float()
                        if self.NUM_GPUS!=0:
                            gt_list = gt_list.cuda(self.device)
                        loss_input.append(gt_list)
                else:
                    if split == ALL:
                        for pred in Ys_pred[source]:
                            loss_input.append(pred.float())
                    else:
                        loss_input.append(Ys_pred[source][split].float())
            # once ready, we have the loss_batch, which is the calculated result after all the 
            # input been fed into the loss module
            loss_batch = loss_func(*loss_input)
            losses_dict[name] = loss_batch

        # Total loss will record the addition result of all the loss together
        total_loss = None
     
        # in case where extra printing is enabled, you can see this printed for more loss information
        if (self.print_train_loss_detail and train) or (self.print_val_loss_detail and not train):
            string = "train: " if train else "val: "
            self.logger.info(string + losses_dict.__str__())

        # weighted (weight=value) sum of loss terms (losses_dict[key])
        for key, value in loss_form.items():
            loss_term = losses_dict[key]
            if total_loss is None:
                total_loss = torch.mul(loss_term, value)
            else:
                total_loss += torch.mul(loss_term, value)

        return total_loss, total_loss.cpu().detach().numpy(), losses_dict