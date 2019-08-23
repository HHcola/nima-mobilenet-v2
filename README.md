# Neural Image Assessment
Implementation of Neural Image Assessment in Keras + Tensorflow with weights for MobileNetV2 model trained on AVA and TID dataset.

# Usage
## Evaluation
There are `evaluater\evaluate_mobilenet_v2*.py` scripts which can be used to evaluate an image using a specific model. The weights for the specific model must be downloaded from the [Releases Tab] and placed in the weights directory.

### Arguments: 
```
-t    : Pass 'ava' or 'tid' dataset as train data.
```

## Training
The AVA dataset is required for training these models. I used 250,000 images to train and the last 5000 images to evaluate .
### Direct-Training
In direct training, you have to ensure that the model can be loaded, trained, evaluated and then saved all on a single GPU.

Use the `train\train_mobilenet_v2.py` scripts for direct training.

# Requirements
- Keras
- Tensorflow (CPU to evaluate, GPU to train)
- Numpy
- Path.py
- PIL
