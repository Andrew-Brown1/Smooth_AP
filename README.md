# Smooth_AP
code for the ECCV '20 paper ["Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval"](https://www.robots.ox.ac.uk/~vgg/research/smooth-ap/)

The PyTorch implementation of the Smooth-AP loss function is found in src/Smooth_AP_loss.py

Training code and pre-trained weights coming soon...

![teaser](https://github.com/Andrew-Brown1/Smooth_AP/blob/master/ims/teaser.png)

## Data
This repository is used for training using Smooth-AP loss on the following datasets:

- PKU Vehicle ID (obtained from this website https://pkuml.org/resources/pku-vehicleid.html - must email authors for download permission)
- INaturalist (2018 version - obtained from this website https://www.kaggle.com/c/inaturalist-2018/data)

We are the first to use the large-scale INaturalist dataset for the task of image retreival. The dataset splits can be downloaded here: https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing . Unpack the zip into the INaturalist dataset directory. 

## Training the model
training results for the Vehicle ID and Inaturalist datasets can be replicated using this repository. To train the model on the Vehicle ID dataset, you can run: 

- python main.py --fc_lr_mul 1 --bs 384

## Paper
If you find this work useful, please consider citing:
```
@InProceedings{Brown20,
  author       = "Andrew Brown and Weidi Xie and Vicky Kalogeiton and Andrew Zisserman ",
  title        = "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval",
  booktitle    = "European Conference on Computer Vision (ECCV), 2020.",
  year         = "2020",
}
```
