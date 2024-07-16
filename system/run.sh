nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data MNIST -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 0 -go cnn > mnist_fedah_dir.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -data Cifar10 -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 0 -go cnn > cifar10_fedah_dir.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 100 -data Cifar100 -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 0 -go cnn > cifar100_fedah_dir.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 200 -data TinyImagenet -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 0 -go cnn > tinyimagenet_fedah_dir.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 200 -data TinyImagenet -m resnet -algo FedAH -gr 2000 -t 2 -eg 5 -did 1 -go resnet > tinyimagenet*_fedah_dir.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 4 -data AGNews -m fastText -algo FedAH -gr 2000 -t 2 -eg 5 -did 1 -go fastText > agnews_fedah_dir.out 2>&1 &

# Scalability-Cifar100
nohup python -u main.py -lbs 16 -nc 10 -jr 1 -nb 100 -data Cifar100_N10 -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 1 -go cnn > cifar100_fedah_dir_n10.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 30 -jr 1 -nb 100 -data Cifar100_N30 -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 1 -go cnn > cifar100_fedah_dir_n30.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 50 -jr 1 -nb 100 -data Cifar100_N50 -m cnn -algo FedAH -gr 2000 -t 1 -eg 5 -did 2 -go cnn > cifar100_fedah_dir_n50.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 100 -jr 1 -nb 100 -data Cifar100_N100 -m cnn -algo FedAH -gr 2000 -t 1 -eg 10 -did 2 -go cnn > cifar100_fedah_dir_n100.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 200 -jr 1 -nb 100 -data Cifar100_N200 -m cnn -algo FedAH -gr 2000 -t 1 -eg 20 -did 2 -go cnn > cifar100_fedah_dir_n200.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 10 -jr 1 -nb 100 -data Cifar100_N10 -m cnn -algo FedALA -gr 2000 -t 2 -eg 5 -did 2 -go cnn > cifar100_fedala_dir_n10.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 30 -jr 1 -nb 100 -data Cifar100_N30 -m cnn -algo FedALA -gr 2000 -t 2 -eg 5 -did 3 -go cnn > cifar100_fedala_dir_n30.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 50 -jr 1 -nb 100 -data Cifar100_N50 -m cnn -algo FedALA -gr 2000 -t 1 -eg 5 -did 3 -go cnn > cifar100_fedala_dir_n50.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 100 -jr 1 -nb 100 -data Cifar100_N100 -m cnn -algo FedALA -gr 2000 -t 1 -eg 10 -did 3 -go cnn > cifar100_fedala_dir_n100.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 200 -jr 1 -nb 100 -data Cifar100_N200 -m cnn -algo FedALA -gr 2000 -t 1 -eg 20 -did 3 -go cnn > cifar100_fedala_dir_n200.out 2>&1 &


# Local epochs-Cifar10
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 5 -data Cifar10 -m cnn -algo FedAH -gr 1000 -t 1 -eg 2 -did 4 -go cnn > cifar10_fedah_dir_ls5.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 10 -data Cifar10 -m cnn -algo FedAH -gr 800 -t 1 -eg 2 -did 4 -go cnn > cifar10_fedah_dir_ls10.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 20 -data Cifar10 -m cnn -algo FedAH -gr 500 -t 1 -eg 2 -did 4 -go cnn > cifar10_fedah_dir_ls20.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 40 -data Cifar10 -m cnn -algo FedAH -gr 300 -t 1 -eg 2 -did 4 -go cnn > cifar10_fedah_dir_ls40.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 5 -data Cifar10 -m cnn -algo FedALA -gr 1000 -t 1 -eg 2 -did 5 -go cnn > cifar10_fedala_dir_ls5.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 10 -data Cifar10 -m cnn -algo FedALA -gr 800 -t 1 -eg 2 -did 5 -go cnn > cifar10_fedala_dir_ls10.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 20 -data Cifar10 -m cnn -algo FedALA -gr 500 -t 1 -eg 2 -did 5 -go cnn > cifar10_fedala_dir_ls20.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 20 -jr 1 -nb 10 -ls 40 -data Cifar10 -m cnn -algo FedALA -gr 300 -t 1 -eg 2 -did 6 -go cnn > cifar10_fedala_dir_ls40.out 2>&1 &


# Join ratios-Cifar100
nohup python -u main.py -lbs 16 -nc 50 -rjr true -jr 0.5 -nb 100 -data Cifar100_N50 -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 6 -go cnn > cifar100_fedah_dir_jr0.5.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 50 -rjr true -jr 0.1 -nb 100 -data Cifar100_N50 -m cnn -algo FedAH -gr 2000 -t 2 -eg 5 -did 6 -go cnn > cifar100_fedah_dir_jr0.1.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 50 -rjr true -jr 0.5 -nb 100 -data Cifar100_N50 -m cnn -algo FedALA -gr 2000 -t 2 -eg 5 -did 7 -go cnn > cifar100_fedala_dir_jr0.5.out 2>&1 &
nohup python -u main.py -lbs 16 -nc 50 -rjr true -jr 0.1 -nb 100 -data Cifar100_N50 -m cnn -algo FedALA -gr 2000 -t 2 -eg 5 -did 7 -go cnn > cifar100_fedala_dir_jr0.1.out 2>&1 &

