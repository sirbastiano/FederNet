# FederNet: a robust convnet-based terrain relative navigation system for planetary applications
## Towards autonomous navigation
### - University of Study of Naples Federico II
![Logo_low](https://user-images.githubusercontent.com/71963566/104814458-43317300-580f-11eb-8def-73051d1459de.png)

The TRN system–that has been called “FederNet”– that combines the strength of a convolutional neural network (CNN or convnet) with the robustness of projective invariants theory. FederNet is specifically designed for a lunar mission but its applicability could be furthed extended to other airless bodies.
The core algorithm is the crater detection algortihm (CDA), an implementation of a version of [Mask R-CNN](https://arxiv.org/abs/1703.06870)(modified by akTwelve) on Python 3, Keras, and TensorFlow 2 for my master thesis. The model generates bounding boxes and segmentation masks for each instance of craters in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository includes:
* Source code of crater detection algorithm (CDA) 
* Source code of crater matching algorithm (CMA) 
* Source code of position estimation algorithm (PEA)
* Implementation of the EKF, through filterpy library
* Pre-trained weights for Lunar DEM LOLA-KAGUYA
* Datasets of training, test and validation (instance segmentation)

## Relator: Prof. Alfredo Renga
## Author: Roberto Del Prete

## Citation
Use this bibtex to cite this repository:
```
@misc{FederNet_TRN_2020,
  title={FederNet: a robust convnet-based terrain relative navigation system for planetary applications},
  author={Roberto Del Prete},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/SirBastiano/Mask_RCNN}},
}

