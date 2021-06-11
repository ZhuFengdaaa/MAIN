![overview](./assets/overview.jpg)

# MAIN: A Real-world Multi-agent Indoor Navigation Benchmark for Cooperative Learning

An implementation of "MAIN: A Real-world Multi-agent Indoor Navigation Benchmark for Cooperative Learning"

<!-- ![](docs/main_visualization.gif) -->

## Architecture Overview

![](assets/framework_v2.jpg)


## Installing dependencies:

This code is tested on python 3.6.10, pytorch v1.4.0 and CUDA V9.1.85.

Install pytorch from https://pytorch.org/ according to your machine configuration.

This code uses older versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab). Install them by running the following commands:

#### Installing habitat-sim:

```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
git checkout ae6ba1cdc772f7a5dedd31cbf9a5b77f6de3ff0f
pip install -r requirements.txt; 
python setup.py install --headless # (for headless machines with GPU)
python setup.py install # (for machines with display attached)
```

#### Installing habitat-lab:
```
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout 676e593b953e2f0530f307bc17b6de66cff2e867
pip install -e .
```

<!-- We know that roadblocks can come up while installing Habitat, we are here to help! For installation issues in habitat, feel free to raise an issue in this repository, or in the corresponding habitat repository. -->

## Setup
Clone the repository and install the requirements:

```
git clone https://github.com/ZhuFengdaaa/MAIN.git
cd MAIN
pip install -r requirements.txt
```

#### Download MAIN dataset

```
mkdir data
cd data
mkdir datasets
cd datasets
```

download dataset from the link below: 

`https://drive.google.com/file/d/1H3fvyPi_OoXJfzV0HzK-Gix5fiizdM4Z/view?usp=sharing`

extract the dataset file: 

`tar-xf main_data.tar`

download the oracle occupancy from the link below: 

https://drive.google.com/file/d/1vDY3Wuc8jLSeGwnb2GSlGUzECTgkKo9k/view?usp=sharing

extract the oracle map file: 

`unzip oracle.zip`

#### Download Matterport3D scenes

The Matterport scene dataset and multiON dataset should be placed in `data` folder under the root directory (`multiON/`) in the following format:

```
MAIN/
  data/
    scene_datasets/
      mp3d/
        1LXtFkjw3qL/
          1LXtFkjw3qL.glb
          1LXtFkjw3qL.navmesh
          ...
    datasets/
      multinav/
        3_ON/
          train/
            ...
          val/
            val.json.gz
        2_ON
          ...
        1_ON
          ...
```				

Download Matterport3D data for Habitat by following the instructions mentioned [here](https://github.com/facebookresearch/habitat-api#data).

### Training

For training an OracleEgoMap (`oracle-ego`) agent, run this from the root directory: 

```
python habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_mamonav.yaml --agent-type oracle-ego --run-type train
```
For other agent types, the `--agent-type` argument would change accordingly. 

### Evaluation

```
python habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_mamonav.yaml --agent-type oracle-ego --run-type eval
```


## Citation
> Fengda Zhu, Siyi Hu, Yi Zhang, Haodong Hong, Yi Zhu, Xiaojun Chang, Xiaodan Liang, 2021. MAIN: A Multi-agent Indoor Navigation Benchmark for Cooperative Learning 

## Acknowledgements
This repository is built upon [Habitat Lab](https://github.com/facebookresearch/habitat-lab), [Habitat Sim](https://github.com/facebookresearch/habitat-lab), [multiON](https://github.com/saimwani/multiON).
