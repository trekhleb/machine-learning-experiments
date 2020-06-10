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

@TODO: Explain why do we need these dependencies.

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
