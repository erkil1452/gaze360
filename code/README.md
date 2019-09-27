# Gaze360: Physically Unconstrained Gaze Estimation in the Wild Dataset

## About

This is code for training and running our Gaze360 model. The usage of this code is for non-commercial research use only. By using this code you agree to terms of the [LICENSE](https://github.com/Erkil1452/gaze360/blob/master/LICENSE.md). If you use our dataset or code cite our [paper](x) as:

 > Petr Kellnhofer*, Adrià Recasens*, Simon Stent, Wojciech Matusik, and Antonio Torralba. “Gaze360: Physically Unconstrained Gaze Estimation in the Wild”. IEEE International Conference on Computer Vision (ICCV), 2019.

```
@inproceedings{gaze360_2019,
    author = {Petr Kellnhofer and Adria Recasens and Simon Stent and Wojciech Matusik and and Antonio Torralba},
    title = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Data
You can obtain the Gaze360 dataset and more information at [http://gaze360.csail.mit.edu](http://gaze360.csail.mit.edu). 

This repository provides already processed txt files with the split for training the Gaze360 model. The txt contains the following information:
* Row 1: Image path
* Row 2-4: Gaze vector

Note that these splits only contain the samples which have available a one second window in the dataset.

## Requriments
The implementation has been tested wihth PyTorch 1.1.0 but it is likely to work on previous version of PyTorch as well.


## Structure

The code consists of
- This readme.
- The training/val/test splits to train the Gaze360 model, as described in the Data section.
- The model and loss definition (model.py)
- A script for training and evaluation of the Gaze360 model (run.py).
- A data loader specific for the Gaze360 dataset (data_loader.py)

## Trained models

The model weights can be downloaded from this [link](http://gaze360.csail.mit.edu/files/gaze360_model.pth.tar)

## Gaze360 in videos
A beta version of the notebook describing how to run Gaze360 on Youtube videos is now [online](https://colab.research.google.com/drive/1AUvmhpHklM9BNt0Mn5DjSo3JRuqKkU4y)!
