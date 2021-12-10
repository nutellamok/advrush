# AdvRush
Official Code for [AdvRush: Searching for Adversarially Robust Neural Architectures](https://openaccess.thecvf.com/content/ICCV2021/html/Mok_AdvRush_Searching_for_Adversarially_Robust_Neural_Architectures_ICCV_2021_paper.html) (ICCV '21)

## Environmental Set-up
```
Python == 3.6.12, PyTorch == 1.2.0, torchvision == 0.4.0
```

## AdvRush Search Process
```
cd advrush && python train_search.py --batch_size 32 --gpu 0 --epochs 60 --a_gamma 0.01 --a_warmup_epochs 0 --w_warmup_epochs 60 --loss_hessian loss_cure
```

## Adversarial Training
```
cd advrush && python adv_train.py --batch_size 64 --gpu 0 --epochs 200 --adv_loss pgd --arch ADVRUSH
```

## Evaluation under PGD Attack
Prior to the evaluation process, add all necessary checkpoint files (preferably in the form of .pth.tar) to the /eval/checkpoints folder.
To conduct white-box attacks, 
```
cd eval &&
python pgd_attack.py --white-box-attack True --test-batch-size 10 --arch [arch_name] --checkpoint [./checkpoints/file_name.pth.tar] --data_type [cifar10/svhn]
```

To conduct black-box attacks, 
```
cd eval &&
python pgd_attack.py --test-batch-size 10 --target_arch [target_arch] --target_checkpoint [./checkpoints/target_file.pth.tar] --source_arch [source_arch] --source_checkpoint [./checkpoints/source_file.pth.tar] --data_type cifar10
```

## References

DARTS: Differentiable Architecture Search [ICLR '19] [code](https://github.com/quark0/darts) [paper](https://arxiv.org/abs/1806.09055)

Robustness via Curvature Regularization, and Vice Versa [CVPR '19] [code](https://github.com/F-Salehi/CURE_robustness) [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper.pdf)

Tradeoff-inspired Adversarial Defense via Surrogate-loss Minimization [ICML '19] [code](https://github.com/yaodongyu/TRADES) [paper](https://arxiv.org/pdf/1901.08573.pdf)
