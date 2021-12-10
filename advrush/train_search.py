import os
import sys
import time
import glob
from random import shuffle
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from regularizer import *
from tensorboardX import SummaryWriter
import hessianflow as hf
import hessianflow.optimizer.optm_utils as hf_optm_utils
import hessianflow.optimizer.progressbar as hf_optm_pgb

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--a_gamma', type=float, default=0.01, help='a regularization strength')
parser.add_argument('--w_gamma', type=float, default=1e-4, help='w regularization strength')
parser.add_argument('--a_warmup_epochs', type=int, default=50, help='num of warm up epochs before using Hessian - architecture weight')
parser.add_argument('--w_warmup_epochs', type=int, default=60, help='num of warm up epochs before using Hessian - model parameters')
parser.add_argument('--loss_hessian', type=str, default='loss_cure', help='type of hessian loss to use, loss_eigen')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if not os.path.isdir(os.path.join(args.save, './log')):
  os.makedirs(os.path.join(args.save, './log'))
tb_logger = SummaryWriter(os.path.join(args.save, './log'))

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))
    h_all = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    h_all = np.append(h_all, [1.5]*int(args.epochs-6))
    # training
    train_acc, train_obj, a_reg, w_reg = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, h=h_all[epoch])
    logging.info('train_acc %f', train_acc)
    tb_logger.add_scalar('train_accuracy', train_acc, epoch)
    tb_logger.add_scalar('train_loss', train_obj, epoch)
    tb_logger.add_scalar('alpha_regularization', a_reg, epoch)
    tb_logger.add_scalar('weight_regularization', w_reg, epoch)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'model_optimizer': optimizer.state_dict(),
        'arch_optimizer': architect.optimizer.state_dict(),
        'model':  model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'alpha_normal': model.alphas_normal,
        'alpha_reduce': model.alphas_reduce}, is_best=False, save=args.save, epoch=epoch)



def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, h):
  objs = utils.AvgrageMeter()
  a_regs = utils.AvgrageMeter()
  w_regs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda(non_blocking=True)
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    a_regularizer = architect.step(input, target, epoch, args.a_warmup_epochs, args.a_gamma, criterion, args.loss_hessian, valid_queue, input_search, target_search, lr, optimizer, unrolled=args.unrolled, h=h)

    optimizer.zero_grad()
    logits = model(input)

    if epoch < args.w_warmup_epochs:
      loss = criterion(logits, target)
      w_regularizer = torch.tensor(0, dtype=torch.float)
    else:
      if args.loss_hessian == 'loss_cure':
        reg = loss_cure(model, criterion, lambda_=1, device='cuda')
        w_regularizer, grad_norm = reg.regularizer(input, target, h=h)
      else:
        reg = loss_eigen(model, train_queue, input, target, criterion, full_eigen=False, maxIter=10, tol=1e-2)
        regularizer, _ = reg.regularizer()

    loss = criterion(logits, target) + args.w_gamma * w_regularizer
    print(f'epoch={epoch} | step={step} | loss={loss} | w_reg={w_regularizer} | a_reg = {a_regularizer}')

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    w_regs.update(w_regularizer.data.item(), n)
    a_regs.update(a_regularizer.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d objs %e a_regs %e w_regs %e %f %f', step, objs.avg, a_regs.avg, w_regs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg, a_regs.avg, w_regs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, requires_grad=False).cuda(non_blocking=True)
      target = Variable(target, requires_grad=False).cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

