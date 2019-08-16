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
- The training splits, as described in the Data section.
- The model and loss definition (model.py)
- A training script (train.py)
- A data loader specific for the Gaze360 dataset (data_loader.py)

## Trained models

The model weights can be downloaded from this [link](http://gaze360.csail.mit.edu/files/gaze360_model.pth.tar)

## Future releases
We are planning to release soon code and instructions to use Gaze360 in arbitrary videos. 