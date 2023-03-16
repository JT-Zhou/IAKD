# IAKD

### Environments:
- Python 3.8
- PyTorch 1.12.0
- torchvision 0.13.0

### Train

1. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./mdistiller/models/cifar/download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for instance, DKD method.
  python3 train.py --cfg configs/cifar100/dkd.yaml
  
  # for instance, our IAKD method.
  python3 train.py --cfg configs/cifar100/IAKD/wrn40_2_res8x4.yaml

  # you can also change settings at command line
  python3 train.py --cfg configs/cifar100/IAKD/wrn40_2_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
  ```
2. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # for instance, our IAKD method.
  python3 train.py --cfg configs/imagenet/r34_r18/iakd.yaml
  ```
  
### Acknowledgement
  Thanks for DKD and ReviewKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller) and the [ReviewKD's codebase](https://github.com/dvlab-research/ReviewKD).
