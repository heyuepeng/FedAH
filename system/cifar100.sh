nohup python -u generate_Cifar100.py noniid - dir Cifar100_N10/ 10 > Cifar100_N10_dataset.out 2>&1 &
nohup python -u generate_Cifar100.py noniid - dir Cifar100_N30/ 30 > Cifar100_N30_dataset.out 2>&1 &
nohup python -u generate_Cifar100.py noniid - dir Cifar100_N50/ 50 > Cifar100_N50_dataset.out 2>&1 &
nohup python -u generate_Cifar100.py noniid - dir Cifar100_N100/ 100 > Cifar100_N100_dataset.out 2>&1 &
nohup python -u generate_Cifar100.py noniid - dir Cifar100_N200/ 200 > Cifar100_N200_dataset.out 2>&1 &