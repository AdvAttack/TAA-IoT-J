# TAA-IoT-J

This repository contains Python implementations of the IEEE IoT-J paper: Targeted Attention Attack on Deep Learning Models in Road Sign Recognition.
<p align="center">
<img src="https://github.com/AdvAttack/TAA-IoT-J/blob/master/image/fig-framework.png" width=100% height=100%>
</p>

## Usage
#### Build the running environment by the following commands:
```
cd TAA_optimization
rm Pipfile
pipenv install tensorflow==1.4.1 (change this to `pipenv install tensorflow-gpu==1.4.1`, if you have a GPU on your system)
pipenv install keras==1.2.0
pipenv install scipy
pipenv install opencv-python
pipenv install pillow
pipenv shell
```
#### Directly run the perturbation optimization code by: 
```
sh Perturbation_optimization_Stop_SpeedLimit45.sh
```
This code will load a well trained CNN model from "models" folder

#### If you want to train the CNN classifier by yourself, run the
```
python Train.py
```
With each retrain, the training/testing accuracy may slight different because it re-select training data and testing data randomly in each run.

#### To get the targeted attack result, run:
```
python Test.py
```

#### Change the load data command of "Test.py" from
```
imgs_test, filenames = load_many_images('./data/test/0')
```
to 
```
imgs_test, filenames = load_many_images('./data/Physical_world_attack/***')
```
The **physical world attack** experiment will be implemented.

## Citation

When using this code, or the ideas of TAA, please cite the following paper
<pre><code>@article{yang2020targeted,
  title={Targeted Attention Attack on Deep Learning Models in Road Sign Recognition},
  author={Yang, Xinghao and Liu, Weifeng and Zhang, Shengli and Liu, Wei and Tao, Dacheng},
  journal={IEEE Internet of Things Journal},
  year={2020},
  publisher={IEEE}
}
</code></pre>
