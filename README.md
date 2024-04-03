[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/UwpqMYOQ)
# e4040-2023fall-project-aaa3-ar4634-ab5640-ap4478
# SWIN Transformer Model
This README provides a concise overview of the SWIN Transformer model, instructions on how to create and compile a model using TensorFlow, and access to weights obtained during experiments.

### Creating and Compiling the SWIN Transformer Model
To create and compile a SWIN Transformer model using TensorFlow, follow the provided code snippet. Ensure that the code is executed within the context of `strategy.scope()` for distributed training, if necessary. Install the required libraries, including TensorFlow and have the swin_transformer.py file in the same directory as the IPYNB, before running the code.

```python
import tensorflow as tf
from swin_transformer import SwinTransformer 
from tensorflow.keras.applications.imagenet_utils import preprocess_input

with strategy.scope():
    img_adjust_layer = tf.keras.layers.Lambda(lambda data: preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMAGE_SIZE, 3])
    pretrained_model = SwinTransformer('swin_large_224', num_classes=len(CLASSES), include_top=False, pretrained=True, use_tpu=False)
    
    model = tf.keras.Sequential([
        img_adjust_layer,
        pretrained_model,
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
            
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

### Access to Model Weights
The weights obtained from the experiments can be accessed through the following Google Drive link: [SWIN Transformer Pre-trained Weights](https://drive.google.com/drive/folders/1AhIsPfEOpU42-BGeHxCGDL5DTq9_xDDM?usp=sharing). Download the weights as needed and use them to initialize your SWIN Transformer model for specific tasks or further experimentation. Ensure compatibility with the model architecture to achieve optimal performance.

### Access the Dataset

#### Data that gave the best results

```python
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path("flower-classification")
IMAGE_SIZE = [224, 224]
EPOCHS = 15
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
```

#### Other datasets

This is just to get the dataset. The remaining functions and data pre-processing steps have been defined in the notebook. The following is a snapshot of the data that we have used in our project and it gave us the best results.

<img width="476" alt="data_sample" src="https://github.com/ecbme4040/e4040-2023fall-project-aaa3/assets/63908462/df052b40-396f-49c1-8f9d-f0146d87bf49">

Other datasets that we have tried to work with are the [Tiny-Imagenet Dataset](https://drive.google.com/file/d/1-tz9PB1dmqVBM1ZYbQ7bW5kvRmeN_dts/view?usp=sharing) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). Since we were loading pre-trained weights, we had to upscale these images to 224 x 224 size, which led to a poor performance.   

### Credits

- Project Inspiration: This project was inspired from the paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- Code implementation: The implementation used in this project is credited to this [implementation](https://github.com/rishigami/Swin-Transformer-TF). We had to follow the exact same structure of the implementation that was there in this repository in order for us to properly load the pre-trained weights, and use them as a starting point to train our model on the "Flower" dataset. This adaptation ensures compatibility and consistency, especially with pre-existing weights, which helped us achieve training and convergence, especially on the limited resources that were available to us.

### File Directory Structure
```
./
|-- figures
|   |-- aaa3_gcp_work_example_screenshot_1.png
|   |-- aaa3_gcp_work_example_screenshot_2.png
|   |-- aaa3_gcp_work_example_screenshot_3.png
|-- swin_transformer.py
|-- swin_notebook.ipynb
|-- README.md
|-- E4040.2023Fall.aaa3.report.ar4634.ab5640.ap4478.pdf
|-- Tiny_Imagenet.ipynb

```
