## cnnMAIN.py

### outline for traning a CNN with Keras. CNNs are interchangeable

This script implements generators to get training data on-the-fly. The custom generators can be used with augmentation scripts and custom file-reading scripts. The CNN that is implemented can be from any file so long as the output is a Keras model object.

The `infuncs3.py` file (https://github.com/mlnotebook/infuncs) has helper functions that get the train/test/validation split from a datafile (`get_split`) and to preprocess and yield the data to a generator, including data augmnetation, (`get_batch`).

Notes
* This is built on the Keras library which performs the model fitting using `fit_generator` - a function that takes in custom generators.
* The generator will automatically randomise the input on each epoch.
* Augmentation is implements in custom functions that require `transforms.py` at https://github.com/mlnotebook.infuncs - but you implement your own so long as the augmenter returns an array of images and an array of labels.

