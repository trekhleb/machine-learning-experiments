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

As you may see, the dataset consists of 3 files. We need to merge information from those 3 files into one dataset later.

Let's download a datasets and preview examples from them.

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

## Preprocessing the dataset

### Filtering out incomplete examples

```python
# Filters out recipes which don't have either title or ingredients or instructions.
def recipe_validate_required_fields(recipe):
    required_keys = ['title', 'ingredients', 'instructions']
    
    if not recipe:
        return False
    
    for required_key in required_keys:
        if not recipe[required_key]:
            return False
        
        if type(recipe[required_key]) == list and len(recipe[required_key]) == 0:
            return False
    
    return True
```

```
dataset_validated = [recipe for recipe in dataset_raw if recipe_validate_required_fields(recipe)]

print('Dataset size BEFORE validation', len(dataset_raw))
print('Dataset size AFTER validation', len(dataset_validated))
print('Number of invalide recipes', len(dataset_raw) - len(dataset_validated))
```

```
Dataset size BEFORE validation 125164
Dataset size AFTER validation 122938
Number of invalide recipes 2226
```

### Converting recipes objects into strings

To help our RNN learn the structure of the text let's add 3 "landmarks" to it. We will use these unique "title", "ingredients" and "instruction" landmarks to separate a logic sections of each recipe.

```python
STOP_WORD_TITLE = '📗 '
STOP_WORD_INGREDIENTS = '\n🥕\n\n'
STOP_WORD_INSTRUCTIONS = '\n📝\n\n'
```

```python
# Converts recipe object to string (sequence of characters) for later usage in RNN input.
def recipe_to_string(recipe):
    # This string is presented as a part of recipes so we need to clean it up.
    noize_string = 'ADVERTISEMENT'
    
    title = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')
    
    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient.replace(noize_string, '')
        if ingredient:
            ingredients_string += f'• {ingredient}\n'
    
    instructions_string = ''
    for instruction in instructions:
        instruction = instruction.replace(noize_string, '')
        if instruction:
            instructions_string += f'▪︎ {instruction}\n'
    
    return f'{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}'
```

```python
dataset_stringified = [recipe_to_string(recipe) for recipe in dataset_validated]

print('Stringified dataset size: ', len(dataset_stringified))
```

```
Stringified dataset size:  122938
```

```python
for recipe_index, recipe_string in enumerate(dataset_stringified[:10]):
    print('Recipe #{}\n---------'.format(recipe_index + 1))
    print(recipe_string)
    print('\n')
```

```
Recipe #1
---------
📗 Slow Cooker Chicken and Dumplings

🥕

• 4 skinless, boneless chicken breast halves 
• 2 tablespoons butter 
• 2 (10.75 ounce) cans condensed cream of chicken soup 
• 1 onion, finely diced 
• 2 (10 ounce) packages refrigerated biscuit dough, torn into pieces 

📝

▪︎ Place the chicken, butter, soup, and onion in a slow cooker, and fill with enough water to cover.
▪︎ Cover, and cook for 5 to 6 hours on High. About 30 minutes before serving, place the torn biscuit dough in the slow cooker. Cook until the dough is no longer raw in the center.



Recipe #2
---------
📗 Awesome Slow Cooker Pot Roast

🥕

• 2 (10.75 ounce) cans condensed cream of mushroom soup 
• 1 (1 ounce) package dry onion soup mix 
• 1 1/4 cups water 
• 5 1/2 pounds pot roast 

📝

▪︎ In a slow cooker, mix cream of mushroom soup, dry onion soup mix and water. Place pot roast in slow cooker and coat with soup mixture.
▪︎ Cook on High setting for 3 to 4 hours, or on Low setting for 8 to 9 hours.



Recipe #3
---------
📗 Brown Sugar Meatloaf

🥕

• 1/2 cup packed brown sugar 
• 1/2 cup ketchup 
• 1 1/2 pounds lean ground beef 
• 3/4 cup milk 
• 2 eggs 
• 1 1/2 teaspoons salt 
• 1/4 teaspoon ground black pepper 
• 1 small onion, chopped 
• 1/4 teaspoon ground ginger 
• 3/4 cup finely crushed saltine cracker crumbs 

📝

▪︎ Preheat oven to 350 degrees F (175 degrees C). Lightly grease a 5x9 inch loaf pan.
▪︎ Press the brown sugar in the bottom of the prepared loaf pan and spread the ketchup over the sugar.
▪︎ In a mixing bowl, mix thoroughly all remaining ingredients and shape into a loaf. Place on top of the ketchup.
▪︎ Bake in preheated oven for 1 hour or until juices are clear.



Recipe #4
---------
📗 Best Chocolate Chip Cookies

🥕

• 1 cup butter, softened 
• 1 cup white sugar 
• 1 cup packed brown sugar 
• 2 eggs 
• 2 teaspoons vanilla extract 
• 3 cups all-purpose flour 
• 1 teaspoon baking soda 
• 2 teaspoons hot water 
• 1/2 teaspoon salt 
• 2 cups semisweet chocolate chips 
• 1 cup chopped walnuts 

📝

▪︎ Preheat oven to 350 degrees F (175 degrees C).
▪︎ Cream together the butter, white sugar, and brown sugar until smooth. Beat in the eggs one at a time, then stir in the vanilla. Dissolve baking soda in hot water. Add to batter along with salt. Stir in flour, chocolate chips, and nuts. Drop by large spoonfuls onto ungreased pans.
▪︎ Bake for about 10 minutes in the preheated oven, or until edges are nicely browned.



Recipe #5
---------
📗 Homemade Mac and Cheese Casserole

🥕

• 8 ounces whole wheat rotini pasta 
• 3 cups fresh broccoli florets 
• 1 medium onion, chopped 
• 3 cloves garlic, minced 
• 4 tablespoons butter, divided 
• 2 tablespoons all-purpose flour 
• 1/4 teaspoon salt 
• 1/8 teaspoon ground black pepper 
• 2 1/2 cups milk 
• 8 ounces Cheddar cheese, shredded 
• 4 ounces reduced-fat cream cheese, cubed and softened 
• 1/2 cup fine dry Italian-seasoned bread crumbs 
• Reynolds Wrap® Non Stick Aluminum Foil 

📝

▪︎ Preheat oven to 350 degrees F. Line a 2-quart casserole dish with Reynolds Wrap(R) Pan Lining Paper, parchment side up. No need to grease dish.
▪︎ Cook the pasta in a large saucepan according to the package directions, adding the broccoli for the last 3 minutes of cooking. Drain. Return to the saucepan and set aside.
▪︎ Cook the onion and garlic in 2 tablespoons hot butter in a large skillet 5 to 7 minutes or until tender. Stir in flour, salt, and black pepper. Add the milk all at once. Cook and stir over medium heat until slightly thickened and bubbly. Add cheddar cheese and cream cheese, stirring until melted. Pour cheese sauce over the pasta and broccoli and stir until well combined.
▪︎ Melt the remaining 2 tablespoons butter and mix with the bread crumbs in a small bowl. Transfer the pasta mixture to the prepared casserole dish. Top with the buttery bread crumbs.
▪︎ Bake, uncovered, about 25 minutes or until bubbly and internal temperature is 165 degrees F. Let stand for 10 minutes before serving.



Recipe #6
---------
📗 Banana Banana Bread

🥕

• 2 cups all-purpose flour 
• 1 teaspoon baking soda 
• 1/4 teaspoon salt 
• 1/2 cup butter 
• 3/4 cup brown sugar 
• 2 eggs, beaten 
• 2 1/3 cups mashed overripe bananas 

📝

▪︎ Preheat oven to 350 degrees F (175 degrees C). Lightly grease a 9x5 inch loaf pan.
▪︎ In a large bowl, combine flour, baking soda and salt. In a separate bowl, cream together butter and brown sugar. Stir in eggs and mashed bananas until well blended. Stir banana mixture into flour mixture; stir just to moisten. Pour batter into prepared loaf pan.
▪︎ Bake in preheated oven for 60 to 65 minutes, until a toothpick inserted into center of the loaf comes out clean. Let bread cool in pan for 10 minutes, then turn out onto a wire rack.



Recipe #7
---------
📗 Chef John's Fisherman's Pie

🥕

• For potato crust: 
• 3 russet potatoes, peeled and cut into chunks 
• 3 tablespoons butter 
• 1 pinch freshly grated nutmeg 
• salt and ground black pepper to taste 
• 1 pinch cayenne pepper, or to taste 
• 1/2 cup milk 
• For the spinach: 
• 2 teaspoons olive oil 
• 12 ounces baby spinach leaves 
• For the sauce: 
• 3 tablespoons butter 
• 3 tablespoons all-purpose flour 
• 2 cloves garlic, minced 
• 2 cups cold milk, divided 
• 2 teaspoons lemon zest 
• For the rest: 
• 1 tablespoon butter 
• salt and ground black pepper to taste 
• 1 pinch cayenne pepper, or to taste 
• 2 pounds boneless cod fillets 
• 1/2 lemon, juiced 
• 1 tablespoon chopped fresh chives for garnish 

📝

▪︎ Bring a large saucepan of salted water and to a boil; add russet potatoes to boiling water and cook until very tender, about 20 minutes. Drain well. Mash in 3 tablespoons butter until thoroughly combined. Season with nutmeg, salt, black pepper, and cayenne pepper to taste. Mash 1/2 cup milk into potato mixture until smooth.
▪︎ Drizzle olive oil in a large Dutch oven over medium-high heat, add spinach, and season with a big pinch of salt. Cook, stirring occasionally, until spinach has wilted, about 1 minute. Transfer to a bowl lined with paper towels to wick away excess moisture.
▪︎ Heat 3 tablespoons butter and flour in a saucepan over medium heat; whisk mixture to a smooth paste. Cook, stirring constantly, until mixture has a nutty smell and is slightly browned, about 2 minutes. Add chopped garlic; whisk until fragrant, 10 to 20 seconds.
▪︎ Whisk 1 cup cold milk into flour mixture; cook until thickened. Whisk in remaining 1 cup milk and lemon zest. Bring white sauce to a gentle simmer, whisking constantly; season with salt. Turn heat to very low and keep sauce warm.
▪︎ Preheat oven to 375 degrees F (190 degrees C). Grease an 8x12-inch casserole dish with 1 tablespoon butter.
▪︎ Season buttered pan with salt, black pepper, and cayenne pepper. Lay boneless cod fillets into the pan in a single layer. Season tops of fillets with more salt, black pepper, and cayenne pepper. Spread spinach evenly over fish and drizzle with lemon juice. Spoon white sauce over spinach; give casserole dish several taps and shakes to eliminate bubbles.
▪︎ Drop mashed potatoes by heaping spoonfuls over the casserole and spread smoothly to cover. Place dish onto a rimmed baking sheet to catch spills.
▪︎ Bake in the preheated oven until bubbling, about 40 minutes. Turn on oven's broiler and broil until potato crust has a golden brown top, about 2 minutes. Fish should flake easily. Let stand 10 minutes before serving. Garnish with a sprinkle of chives.



Recipe #8
---------
📗 Mom's Zucchini Bread

🥕

• 3 cups all-purpose flour 
• 1 teaspoon salt 
• 1 teaspoon baking soda 
• 1 teaspoon baking powder 
• 1 tablespoon ground cinnamon 
• 3 eggs 
• 1 cup vegetable oil 
• 2 1/4 cups white sugar 
• 3 teaspoons vanilla extract 
• 2 cups grated zucchini 
• 1 cup chopped walnuts 

📝

▪︎ Grease and flour two 8 x 4 inch pans. Preheat oven to 325 degrees F (165 degrees C).
▪︎ Sift flour, salt, baking powder, soda, and cinnamon together in a bowl.
▪︎ Beat eggs, oil, vanilla, and sugar together in a large bowl. Add sifted ingredients to the creamed mixture, and beat well. Stir in zucchini and nuts until well combined. Pour batter into prepared pans.
▪︎ Bake for 40 to 60 minutes, or until tester inserted in the center comes out clean. Cool in pan on rack for 20 minutes. Remove bread from pan, and completely cool.



Recipe #9
---------
📗 The Best Rolled Sugar Cookies

🥕

• 1 1/2 cups butter, softened 
• 2 cups white sugar 
• 4 eggs 
• 1 teaspoon vanilla extract 
• 5 cups all-purpose flour 
• 2 teaspoons baking powder 
• 1 teaspoon salt 

📝

▪︎ In a large bowl, cream together butter and sugar until smooth. Beat in eggs and vanilla. Stir in the flour, baking powder, and salt. Cover, and chill dough for at least one hour (or overnight).
▪︎ Preheat oven to 400 degrees F (200 degrees C). Roll out dough on floured surface 1/4 to 1/2 inch thick. Cut into shapes with any cookie cutter. Place cookies 1 inch apart on ungreased cookie sheets.
▪︎ Bake 6 to 8 minutes in preheated oven. Cool completely.



Recipe #10
---------
📗 Singapore Chili Crabs

🥕

• Sauce: 
• 1/2 cup ketchup 
• 1/2 cup chicken broth 
• 1 large egg 
• 2 tablespoons soy sauce 
• 2 tablespoons chile-garlic sauce (such as sambal oelek) 
• 1 tablespoon oyster sauce 
• 1 tablespoon tamarind paste 
• 2 teaspoons fish sauce 
• 2 teaspoons palm sugar 
• 1/4 cup minced shallot 
• 6 cloves garlic, minced 
• 2 tablespoons vegetable oil, or more as needed 
• 2 tablespoons minced fresh ginger root 
• 1 tablespoon minced serrano pepper 
• 2 cooked Dungeness crabs, cleaned and cracked 
• 2 tablespoons chopped fresh cilantro 
• 2 tablespoons sliced green onion (green part only) 

📝

▪︎ Whisk ketchup, chicken broth, egg, soy sauce, chile-garlic sauce, oyster sauce, tamarind paste, fish sauce, and palm sugar together in a bowl.
▪︎ Stir shallots, garlic, oil, ginger, and serrano pepper together in a pot over medium-high heat. Saute until sizzling, about 2 minutes. Add crab to pot, cover the pot with a lid, and shake until crab is completely coated in shallot mixture. Remove lid and cook and stir until heated through, about 3 minutes.
▪︎ Pour ketchup mixture into pot, reduce heat to medium, and cook and stir until sauce thickens and crab is hot about 5 minutes. Remove from heat; stir in cilantro and green onions.
```

```python
print(dataset_stringified[50000])
```

```
📗 Herbed Bean Ragoût 

🥕

• 6 ounces haricots verts (French thin green beans), trimmed and halved crosswise
• 1 (1-pound) bag frozen edamame (soybeans in the pod) or 1 1/4 cups frozen shelled edamame, not thawed
• 2/3 cup finely chopped onion
• 2 garlic cloves, minced
• 1 Turkish bay leaf or 1/2 California bay leaf
• 2 (3-inch) fresh rosemary sprigs
• 1/2 teaspoon salt
• 1/4 teaspoon black pepper
• 1 tablespoon olive oil
• 1 medium carrot, cut into 1/8-inch dice
• 1 medium celery rib, cut into 1/8-inch dice
• 1 (15- to 16-ounces) can small white beans, rinsed and drained
• 1 1/2 cups chicken stock or low-sodium broth
• 2 tablespoons unsalted butter
• 2 tablespoons finely chopped fresh flat-leaf parsley
• 1 tablespoon finely chopped fresh chervil (optional)
• Garnish: fresh chervil sprigs

📝

▪︎ Cook haricots verts in a large pot of boiling salted water until just tender, 3 to 4 minutes. Transfer with a slotted spoon to a bowl of ice and cold water, then drain. Add edamame to boiling water and cook 4 minutes. Drain in a colander, then rinse under cold water. If using edamame in pods, shell them and discard pods. Cook onion, garlic, bay leaf, rosemary, salt, and pepper in oil in a 2- to 4-quart heavy saucepan over moderately low heat, stirring, until softened, about 3 minutes. Add carrot and celery and cook, stirring, until softened, about 3 minutes. Add white beans and stock and simmer, covered, stirring occasionally, 10 minutes. Add haricots verts and edamame and simmer, uncovered, until heated through, 2 to 3 minutes. Add butter, parsley, and chervil (if using) and stir gently until butter is melted. Discard bay leaf and rosemary sprigs.
▪︎ Cook haricots verts in a large pot of boiling salted water until just tender, 3 to 4 minutes. Transfer with a slotted spoon to a bowl of ice and cold water, then drain.
▪︎ Add edamame to boiling water and cook 4 minutes. Drain in a colander, then rinse under cold water. If using edamame in pods, shell them and discard pods.
▪︎ Cook onion, garlic, bay leaf, rosemary, salt, and pepper in oil in a 2- to 4-quart heavy saucepan over moderately low heat, stirring, until softened, about 3 minutes. Add carrot and celery and cook, stirring, until softened, about 3 minutes.
▪︎ Add white beans and stock and simmer, covered, stirring occasionally, 10 minutes. Add haricots verts and edamame and simmer, uncovered, until heated through, 2 to 3 minutes. Add butter, parsley, and chervil (if using) and stir gently until butter is melted. Discard bay leaf and rosemary sprigs.
```

### Filtering out large receipts

Recipes have different lengths. We need to have one hard-coded sequence length limit before feeding recipes sequences to RNN. We need to find out what recipe length will cover most of recipe use-cases and at the same time we want to keep it as small as possible for training performance.

```python
recipes_lengths = []
for recipe_text in dataset_stringified:
    recipes_lengths.append(len(recipe_text))
```

```python
plt.hist(recipes_lengths, bins=50)
plt.show()
```

![Recipes lengths 1](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/recipes_generation/recipes-length-1.png)

```python
plt.hist(recipes_lengths, range=(0, 8000), bins=50)
plt.show()
```

![Recipes lengths 2](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/recipes_generation/recipes-length-2.png)
