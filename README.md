# Code for "Leveraging Self-Supervision for Cross-Domain Crowd Counting" (CVPR 2022)

This repository is a PyTorch implementation for the paper **Leveraging Self-Supervision for Cross-Domain Crowd Counting**, which is accepted as **oral** presentation at CVPR 2022. If you use this code in your research, please cite
[the paper](https://arxiv.org/pdf/2103.16291.pdf).



## Abstract
State-of-the-art methods for counting people in crowded scenes rely on deep networks to estimate crowd density. While effective, these data-driven approaches rely on large amount of data annotation to achieve good performance, which stops these models from being deployed in emergencies during which data annotation is either too costly or cannot be obtained fast enough.

One popular solution is to use synthetic data for training. Unfortunately, due to domain shift, the resulting models generalize poorly on real imagery. We remedy this shortcoming by training with both synthetic images, along with their associated labels, and unlabeled real images. To this end, we force our network to learn perspective-aware features by training it to recognize upside-down real images from regular ones and incorporate into it the ability to predict its own uncertainty so that it can generate useful pseudo labels for fine-tuning purposes. This yields an algorithm that consistently outperforms state-of-the-art cross-domain crowd counting ones without any extra computation at inference time.


![](./images/model.jpg)
Figure 1: **Two-stage approach**. **Top:** During the first training stage, we use synthetic images, real images, and flipped versions of the latter. The network is trained to output the correct people density for the synthetic images and to classify the real images as being flipped or not. **Bottom:** During the second training stage, we use synthetic and real images. We run the previously trained network on the real images and treat the least uncertain people density estimates as pseudo labels. We then fine tune the network on both kinds of images and iterate the process.

## Installation

Please refer to this [page](https://github.com/nikitadurasov/masksembles) for the installation of [Masksembles](https://arxiv.org/abs/2012.08334).

For other packages, please refer to this [page](https://github.com/weizheliu/People-Flows).


## Implementation

Please refer to model.py

## How to Use
The code is managed in the same way as my previous work, please refer to this [code](https://github.com/weizheliu/People-Flows) for detailed information. You only need to replace the "model.py" file with the one attached here and update other info(model name, dataset etc.) accordingly.



 
## Citing

``` 
@InProceedings{Liu_2022_CVPR,

author = {Liu, Weizhe and Durasov, Nikita and Fua, Pascal},

title = {Leveraging Self-Supervision for Cross-Domain Crowd Counting},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

month = {June},

year = {2022}

}

``` 

## Note

I already left EPFL and therefore I'm not able to reach my previous code/data. This is a quick implementation based on my memory and I do not even have the machine/data to test it, if you find any bug/typo with this code, please let me know.


## Contact

For any questions regard this paper/code, please directly contact [Weizhe Liu](mailto:weizheliu1991@163.ch).

