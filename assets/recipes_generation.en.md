# 🏋🏻‍ I've trained Recurrent Neural Network to generate recipes, and it suggested me to cook 🥤 Cream Soda with 🧅 Onions

## TL;DR

I've trained Recurrent Neural Network (RNN) on _~100k_ recipes using [TensorFlow](https://www.tensorflow.org/). Here is what I ended up with:

- 🎨 [Recipes generator demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/RecipeGenerationRNN)
- 🏋🏻‍ [Model training process](https://github.com/trekhleb/machine-learning-experiments/blob/master/experiments/recipe_generation_rnn/recipe_generation_rnn.ipynb)

This article contains details of model training with TensorFlow code examples (Python).

![Recipe generator demo](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/recipes_generation/demp.gif)

## Experiment overview

I'm a complete beginner in Machine Learning 👶🏻. Creating a _Recipe Generator_ was yet another attempt to play around with _Recurrent Neural Networks_ in my [🤖 Interactive Machine Learning Experiments](https://github.com/trekhleb/machine-learning-experiments) repository and to learn how those RNNs are implemented in TensorFlow in particular.

I was curious to see:

- what weird recipes names and ingredients combination the RNN would suggest
- will it learn that each recipe consists of several blocks (name, ingredients, cooking instructions)
- will it learn English grammar and punctuation from scratch just for several hours of training (I wish I had this skill)

I decided to experiment with **character-based RNN** this time. It means that I will not teach my RNN to understand words and sentences but rather to understand letters and theirs sequences.




@TODO: Explain why do we need these dependencies.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
import json
import zipfile
```

```python
print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
```

```
Python version: 3.7.6
Tensorflow version: 2.1.0
Keras version: 2.2.4-tf
```

```python
# Create cache folder.
cache_dir = './tmp'
pathlib.Path(cache_dir).mkdir(exist_ok=True)
```

```python
# Download and unpack the dataset.
dataset_file_name = 'recipes_raw.zip'
dataset_file_origin = 'https://storage.googleapis.com/recipe-box/recipes_raw.zip'

dataset_file_path = tf.keras.utils.get_file(
    fname=dataset_file_name,
    origin=dataset_file_origin,
    cache_dir=cache_dir,
    extract=True,
    archive_format='zip'
)

print(dataset_file_path)
```

```
./tmp/datasets/recipes_raw.zip
```

```bash
!ls -la ./tmp/datasets/
```

```
total 521128
drwxr-xr-x  7 trekhleb  staff       224 May 13 18:10 .
drwxr-xr-x  4 trekhleb  staff       128 May 18 18:00 ..
-rw-r--r--  1 trekhleb  staff     20437 May 20 06:46 LICENSE
-rw-r--r--  1 trekhleb  staff  53355492 May 13 18:10 recipes_raw.zip
-rw-r--r--  1 trekhleb  staff  49784325 May 20 06:46 recipes_raw_nosource_ar.json
-rw-r--r--  1 trekhleb  staff  61133971 May 20 06:46 recipes_raw_nosource_epi.json
-rw-r--r--  1 trekhleb  staff  93702755 May 20 06:46 recipes_raw_nosource_fn.json
```
