#! /bin/bash
source ./avg.sh VGG16_cifar10 /mnt/disk_data/yaelf/saved/models/VGG16_Cifar10/0715_151237/checkpoint-epoch100.pth l1
source ./avg.sh VGG16_cifar10 /mnt/disk_data/yaelf/saved/models/VGG16_Cifar10/0715_151237/checkpoint-epoch100.pth random
source ./avg.sh VGG16_cifar10 /mnt/disk_data/yaelf/saved/models/VGG16_Cifar10/0715_151237/checkpoint-epoch100.pth lastN

source ./avg.sh VGG16_cifar100 /mnt/disk_data/yaelf/saved/models/VGG16_Cifar100/0621_211610/checkpoint-epoch150.pth l1
source ./avg.sh VGG16_cifar100 /mnt/disk_data/yaelf/saved/models/VGG16_Cifar100/0621_211610/checkpoint-epoch150.pth random
source ./avg.sh VGG16_cifar100 /mnt/disk_data/yaelf/saved/models/VGG16_Cifar100/0621_211610/checkpoint-epoch150.pth lastN

source ./avg.sh InceptionV3_cifar10 /mnt/disk_data/yaelf/saved/models/InceptionV3_Cifar10/0717_220214/checkpoint-epoch100.pth l1
source ./avg.sh InceptionV3_cifar10 /mnt/disk_data/yaelf/saved/models/InceptionV3_Cifar10/0717_220214/checkpoint-epoch100.pth random
source ./avg.sh InceptionV3_cifar10 /mnt/disk_data/yaelf/saved/models/InceptionV3_Cifar10/0717_220214/checkpoint-epoch100.pth lastN

source ./avg.sh InceptionV3_cifar100 /mnt/disk_data/yaelf/saved/models/InceptionV3_Cifar100/0627_151155/checkpoint-epoch50.pth l1
source ./avg.sh InceptionV3_cifar100 /mnt/disk_data/yaelf/saved/models/InceptionV3_Cifar100/0627_151155/checkpoint-epoch50.pth random
source ./avg.sh InceptionV3_cifar100 /mnt/disk_data/yaelf/saved/models/InceptionV3_Cifar100/0627_151155/checkpoint-epoch50.pth lastN

source ./avg.sh Resnet50_cifar10 /mnt/disk_data/yaelf/saved/models/Resnet50_Cifar10/0718_182543/checkpoint-epoch100.pth l1
source ./avg.sh Resnet50_cifar10 /mnt/disk_data/yaelf/saved/models/Resnet50_Cifar10/0718_182543/checkpoint-epoch100.pth random
source ./avg.sh Resnet50_cifar10 /mnt/disk_data/yaelf/saved/models/Resnet50_Cifar10/0718_182543/checkpoint-epoch100.pth lastN

source ./avg.sh Resnet50_cifar100 /mnt/disk_data/yaelf/saved/models/Resnet50_Cifar100/0621_222822/checkpoint-epoch100.pth l1
source ./avg.sh Resnet50_cifar100 /mnt/disk_data/yaelf/saved/models/Resnet50_Cifar100/0621_222822/checkpoint-epoch100.pth random
source ./avg.sh Resnet50_cifar100 /mnt/disk_data/yaelf/saved/models/Resnet50_Cifar100/0621_222822/checkpoint-epoch100.pth lastN

source ./avg.sh EfficientNetB7_cifar10 /mnt/disk_data/yaelf/saved/models/EfficientNetB7_Cifar10/0625_232718/checkpoint-epoch150.pth l1
source ./avg.sh EfficientNetB7_cifar10 /mnt/disk_data/yaelf/saved/models/EfficientNetB7_Cifar10/0625_232718/checkpoint-epoch150.pth random
source ./avg.sh EfficientNetB7_cifar10 /mnt/disk_data/yaelf/saved/models/EfficientNetB7_Cifar10/0625_232718/checkpoint-epoch150.pth lastN

source ./avg.sh EfficientNetB7_cifar100 /mnt/disk_data/yaelf/saved/models/EfficientNetB7_Cifar100/0624_174406/checkpoint-epoch100.pth l1
source ./avg.sh EfficientNetB7_cifar100 /mnt/disk_data/yaelf/saved/models/EfficientNetB7_Cifar100/0624_174406/checkpoint-epoch100.pth random
source ./avg.sh EfficientNetB7_cifar100 /mnt/disk_data/yaelf/saved/models/EfficientNetB7_Cifar100/0624_174406/checkpoint-epoch100.pth lastN
