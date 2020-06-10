# 🏋🏻‍ I've trained Recurrent Neural Network to generate recipes, and it suggested me to cook 🥤 Cream Soda with 🧅 Onions

# A step-by-step guide on how to generate recipes using TensorFlow and Recurrent Neural Network

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




In this experiment we will use character-based [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) to generate cooking recipes. We will try to teach our RNN to generate recipe _name_, _ingredients_ and _cooking instructions_ for us.

I don't expect the RNN to do a strong connection between list of ingredients and cooking instructions but I do expect RNN to learn English grammar and punctuation in couple of hours and to generate some meaningful recipe names along with real food ingredients and cooking instructions.

For this experiment we will use [Tensorflow v2](https://www.tensorflow.org/) with its [Keras API](https://www.tensorflow.org/guide/keras).

⚠️ _The recipes in this notebook are generated just for fun and for learning purposes. The recipes are **not** for actual cooking!_

![recipe_generation_rnn.jpg](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/recipes_generation/cover.jpg)

Photo source: 🥦[home_full_of_recipes](https://www.instagram.com/home_full_of_recipes/)

## Exploring datasets

Let's go through several available dataset and explore their pros and cons. One of the requirement I want the dataset to meet is that it should have not only a list of ingredients but also a cooking instruction. I also want it to have a measures and quantities of each ingredient.

- 🤷 [Recipe Ingredients Dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset/home) _(doesn't have ingredients proportions)_
- 🤷 [Recipe1M+](http://pic2recipe.csail.mit.edu/) _(requires registration to download)_
- 🤷 [Epicurious - Recipes with Rating and Nutrition](https://www.kaggle.com/hugodarwood/epirecipes?select=full_format_recipes.json) _(~20k recipes only, it would be nice to find more)_
- 👍🏻 [**Recipe box**](https://eightportions.com/datasets/Recipes/) _(~125,000 recipes with ingredients proportions, good)_

## Importing dependencies

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

## Loading the dataset

Let's load the dataset using [tf.keras.utils.get_file](https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file). Using `get_file()` utility is convenient because it handles caching for you out of the box. It means that you will download the dataset files only once and then even if you launch the same code block once again it will use cache and the download code will be executed faster.

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


```python
def load_dataset(silent=False):
    dataset_file_names = [
        'recipes_raw_nosource_ar.json',
        'recipes_raw_nosource_epi.json',
        'recipes_raw_nosource_fn.json',
    ]
    
    dataset = []

    for dataset_file_name in dataset_file_names:
        dataset_file_path = f'{cache_dir}/datasets/{dataset_file_name}'

        with open(dataset_file_path) as dataset_file:
            json_data_dict = json.load(dataset_file)
            json_data_list = list(json_data_dict.values())
            dict_keys = [key for key in json_data_list[0]]
            dict_keys.sort()
            dataset += json_data_list

            if silent == False:
                print(dataset_file_path)
                print('===========================================')
                print('Number of examples: ', len(json_data_list), '\n')
                print('Example object keys:\n', dict_keys, '\n')
                print('Example object:\n', json_data_list[0], '\n')
                print('Required keys:\n')
                print('  title: ', json_data_list[0]['title'], '\n')
                print('  ingredients: ', json_data_list[0]['ingredients'], '\n')
                print('  instructions: ', json_data_list[0]['instructions'])
                print('\n\n')

    return dataset  
```

```python
dataset_raw = load_dataset() 
```

```
./tmp/datasets/recipes_raw_nosource_ar.json
===========================================
Number of examples:  39802 

Example object keys:
 ['ingredients', 'instructions', 'picture_link', 'title'] 

Example object:
 {'title': 'Slow Cooker Chicken and Dumplings', 'ingredients': ['4 skinless, boneless chicken breast halves ADVERTISEMENT', '2 tablespoons butter ADVERTISEMENT', '2 (10.75 ounce) cans condensed cream of chicken soup ADVERTISEMENT', '1 onion, finely diced ADVERTISEMENT', '2 (10 ounce) packages refrigerated biscuit dough, torn into pieces ADVERTISEMENT', 'ADVERTISEMENT'], 'instructions': 'Place the chicken, butter, soup, and onion in a slow cooker, and fill with enough water to cover.\nCover, and cook for 5 to 6 hours on High. About 30 minutes before serving, place the torn biscuit dough in the slow cooker. Cook until the dough is no longer raw in the center.\n', 'picture_link': '55lznCYBbs2mT8BTx6BTkLhynGHzM.S'} 

Required keys:

  title:  Slow Cooker Chicken and Dumplings 

  ingredients:  ['4 skinless, boneless chicken breast halves ADVERTISEMENT', '2 tablespoons butter ADVERTISEMENT', '2 (10.75 ounce) cans condensed cream of chicken soup ADVERTISEMENT', '1 onion, finely diced ADVERTISEMENT', '2 (10 ounce) packages refrigerated biscuit dough, torn into pieces ADVERTISEMENT', 'ADVERTISEMENT'] 

  instructions:  Place the chicken, butter, soup, and onion in a slow cooker, and fill with enough water to cover.
Cover, and cook for 5 to 6 hours on High. About 30 minutes before serving, place the torn biscuit dough in the slow cooker. Cook until the dough is no longer raw in the center.




./tmp/datasets/recipes_raw_nosource_epi.json
===========================================
Number of examples:  25323 

Example object keys:
 ['ingredients', 'instructions', 'picture_link', 'title'] 

Example object:
 {'ingredients': ['12 egg whites', '12 egg yolks', '1 1/2 cups sugar', '3/4 cup rye whiskey', '12 egg whites', '3/4 cup brandy', '1/2 cup rum', '1 to 2 cups heavy cream, lightly whipped', 'Garnish: ground nutmeg'], 'picture_link': None, 'instructions': 'Beat the egg whites until stiff, gradually adding in 3/4 cup sugar. Set aside. Beat the egg yolks until they are thick and pale and add the other 3/4 cup sugar and stir in rye whiskey. Blend well. Fold the egg white mixture into the yolk mixture and add the brandy and the rum. Beat the mixture well. To serve, fold the lightly whipped heavy cream into the eggnog. (If a thinner mixture is desired, add the heavy cream unwhipped.) Sprinkle the top of the eggnog with the nutmeg to taste.\nBeat the egg whites until stiff, gradually adding in 3/4 cup sugar. Set aside. Beat the egg yolks until they are thick and pale and add the other 3/4 cup sugar and stir in rye whiskey. Blend well. Fold the egg white mixture into the yolk mixture and add the brandy and the rum. Beat the mixture well. To serve, fold the lightly whipped heavy cream into the eggnog. (If a thinner mixture is desired, add the heavy cream unwhipped.) Sprinkle the top of the eggnog with the nutmeg to taste.', 'title': 'Christmas Eggnog '} 

Required keys:

  title:  Christmas Eggnog  

  ingredients:  ['12 egg whites', '12 egg yolks', '1 1/2 cups sugar', '3/4 cup rye whiskey', '12 egg whites', '3/4 cup brandy', '1/2 cup rum', '1 to 2 cups heavy cream, lightly whipped', 'Garnish: ground nutmeg'] 

  instructions:  Beat the egg whites until stiff, gradually adding in 3/4 cup sugar. Set aside. Beat the egg yolks until they are thick and pale and add the other 3/4 cup sugar and stir in rye whiskey. Blend well. Fold the egg white mixture into the yolk mixture and add the brandy and the rum. Beat the mixture well. To serve, fold the lightly whipped heavy cream into the eggnog. (If a thinner mixture is desired, add the heavy cream unwhipped.) Sprinkle the top of the eggnog with the nutmeg to taste.
Beat the egg whites until stiff, gradually adding in 3/4 cup sugar. Set aside. Beat the egg yolks until they are thick and pale and add the other 3/4 cup sugar and stir in rye whiskey. Blend well. Fold the egg white mixture into the yolk mixture and add the brandy and the rum. Beat the mixture well. To serve, fold the lightly whipped heavy cream into the eggnog. (If a thinner mixture is desired, add the heavy cream unwhipped.) Sprinkle the top of the eggnog with the nutmeg to taste.



./tmp/datasets/recipes_raw_nosource_fn.json
===========================================
Number of examples:  60039 

Example object keys:
 ['ingredients', 'instructions', 'picture_link', 'title'] 

Example object:
 {'instructions': 'Toss ingredients lightly and spoon into a buttered baking dish. Top with additional crushed cracker crumbs, and brush with melted butter. Bake in a preheated at 350 degrees oven for 25 to 30 minutes or until delicately browned.', 'ingredients': ['1/2 cup celery, finely chopped', '1 small green pepper finely chopped', '1/2 cup finely sliced green onions', '1/4 cup chopped parsley', '1 pound crabmeat', '1 1/4 cups coarsely crushed cracker crumbs', '1/2 teaspoon salt', '3/4 teaspoons dry mustard', 'Dash hot sauce', '1/4 cup heavy cream', '1/2 cup melted butter'], 'title': "Grammie Hamblet's Deviled Crab", 'picture_link': None} 

Required keys:

  title:  Grammie Hamblet's Deviled Crab 

  ingredients:  ['1/2 cup celery, finely chopped', '1 small green pepper finely chopped', '1/2 cup finely sliced green onions', '1/4 cup chopped parsley', '1 pound crabmeat', '1 1/4 cups coarsely crushed cracker crumbs', '1/2 teaspoon salt', '3/4 teaspoons dry mustard', 'Dash hot sauce', '1/4 cup heavy cream', '1/2 cup melted butter'] 

  instructions:  Toss ingredients lightly and spoon into a buttered baking dish. Top with additional crushed cracker crumbs, and brush with melted butter. Bake in a preheated at 350 degrees oven for 25 to 30 minutes or until delicately browned.
```

```python
print('Total number of raw examples: ', len(dataset_raw))
```

```
Total number of raw examples:  125164
```
