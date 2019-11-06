# Haemorrhage prediction

This challenge requires us to perform a multilabel classification task to classify 3 types of intracranial haemorrhages from CT scans of the brain. The images are 128x128 and each of them depict a random coronal section of the brain. We modify the problem slightly to add the lack of a haemorrhage as a label which we call `normal`  giving us 4 labels. We prototype with a pretrained `resnet34` architechture using the `fastai` library built on top of `pytorch`. Once we have made all the necessary decision making, we build the final using a pretrained `densenet169`.

Steps to reproduce:

1. Install the environment

`conda env create -f environment.yml`

2. Download the data from https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview. This repository only uses a subset of this data.

3. Run the Jupyter notebook


This is a difficult problem due to the fact that we only have one 2D image of the brain for each haemorrhage. Furthermore, our images are of a very small size. Nevertheless, our final model performs reasonably well. Future steps to improve the model:

- 3D images with more resolution
- Train for longer
- Perform more augmentation using transformations
- Deeper architecture
- Use a larger dataset
- Convert black and white to colour images (this can sometimes help given that we are training on top of ImageNet which are all colour images) or otherwise account for this difference
- Pretrain on other medical images (rather than ImageNet)
- Remove the skull from the images (this technique is sometimes used but we decided not to try this both for simplicity as well due the fact that a survey of the literature suggested possible fractures in the skull could help improve our predictive power) 
