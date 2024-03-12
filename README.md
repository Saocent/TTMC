# TTMC
source code for paper：Triple-task mutual consistency for semi-supervised 3D medical image segmentation（CIBM2024）


## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Usage

1. Clone the repo:
```
git clone https://github.com/Saocent/TTMC.git 
cd TTMC
```
2. Put the data in [data/2018LA_Seg_Training Set].

3. Train the model
```
cd code
python train_LA.py
```

4. Test the model
```
python test_LA.py
```
Our pre-trained models are saved in the model dir. The other comparison method can be found in [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Citation

If you find our work is useful for you, please cite us.
