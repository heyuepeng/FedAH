# FedAH: Aggregated Head for Personalized Federated Learning

![fedah](./fedah.png)



![fedah_client](./fedah_client.png)



This project [FedAH](https://github.com/heyuepeng/FedAH) is based on the open source project [PFLlib](https://github.com/TsingZ0/PFLlib) development.

[![License: GPL v2](https://camo.githubusercontent.com/1b537d3212c421e0362b9c7168f1febd83941d79e8ccd8487309a4a759f7da11/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d47504c5f76322d626c75652e737667)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) [![arXiv](https://camo.githubusercontent.com/ca1d27a07f5525f7d380d41c70a877b7af62e55c39c1199e15129854ab949391/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d323331322e30343939322d6233316231622e737667)](https://arxiv.org/abs/2312.04992)

![img](https://github.com/heyuepeng/PFLlibVSP/raw/main/structure.png)(https://github.com/heyuepeng/PFLlibVSP/blob/main/structure.png)



## Environments

Install [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive).

Install [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda.

```
conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match CUDA version
```



## How to start simulating (examples for FedAH)

- Create proper environments (see [Environments](https://github.com/heyuepeng/FedAH#environments)).

- Download [this project](https://github.com/heyuepeng/FedAH) to an appropriate location using [git](https://git-scm.com/).

  ```
  git clone https://github.com/heyuepeng/FedAH.git
  ```

- Run evaluation:

  ```
  cd ./system
  python main.py -data MNIST -m cnn -algo FedAH -gr 2000 -did 0 # using the MNIST dataset, the FedAH algorithm, and the 4-layer CNN model
  ```

  Or you can uncomment the lines you need in `./system/examples.sh` and run:

  ```
  cd ./system
  sh examples.sh
  ```

**Note**: The hyper-parameters have not been tuned for the algorithms. The values in `./system/examples.sh` are just examples. You need to tune the hyper-parameters by yourself.
