# 🤖 Interactive Machine Learning Experiments

This is a collection of interactive machine-learning experiments. Each experiment consists of 🏋️ Jupyter/Colab _notebook_ (to see how model was trained) and 🎨 _demo page_ (to see a model in action right in your browser).

- 🎨 [Launch ML experiments demo](http://trekhleb.github.io/machine-learning-experiments)
- 🏋️ [Launch ML experiments Jupyter notebooks](https://nbviewer.jupyter.org/github/trekhleb/machine-learning-experiments/tree/master/experiments/)

<hr/>

> ⚠️ This repository contains machine learning **experiments** and **not** a production ready, reusable, optimised and fine-tuned code and models. This is rather a sandbox or a playground for learning and trying different machine learning approaches, algorithms and data-sets.

## Experiments

### Neural Networks (NN)

<table>
  <thead>
    <tr>
      <th align="left" width="150" style="width: 150px !important"> </th>
      <th align="left" width="200" style="width: 200px !important">Experiment</th>
      <th align="left" width="140" style="width: 140px !important">Model training</th>
      <th align="left">Tags</th>
      <th align="left">Dataset</th>
    </tr>
  </thead>
  <tbody>
    <!-- Experiment -->
    <tr>
      <td>
        <img src="assets/images/digits_recognition_mlp.png" alt="Handwritten digits recognition (MLP)" width="150" />
      </td>
      <td>
        <b>Handwritten digits recognition (MLP)</b>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/DigitsRecognitionMLP">
          <img src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&message=Demo&color=green" alt="Launch demo">
        </a>
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <code>Multilayer&nbsp;Perceptron</code>,
        <code>MLP</code>,
        <code>Tensorflow</code>,
        <code>Keras</code>
      </td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/mnist">
          MNIST
        </a>
      </td>
    </tr>
  </tbody>
</table>

### Convolutional Neural Networks (CNN)

<table>
  <thead>
    <tr>
      <th align="left" width="150" style="width: 150px !important"> </th>
      <th align="left" width="200" style="width: 200px !important">Experiment</th>
      <th align="left" width="140" style="width: 140px !important">Model demo & training</th>
      <th align="left">Tags</th>
      <th align="left">Dataset</th>
    </tr>
  </thead>
  <tbody>
    <!-- Experiment -->
    <tr>
      <td>
        <img src="assets/images/digits_recognition_cnn.png" alt="Handwritten digits recognition (CNN)" />
      </td>
      <td>
        <b>Handwritten digits recognition (CNN)</b>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/DigitsRecognitionCNN">
          <img src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&message=Demo&color=green" alt="Launch demo">
        </a>
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <code>Convolutional&nbsp;Neural&nbsp;Network</code>,
        <code>CNN</code>,
        <code>Tensorflow</code>,
        <code>Keras</code>
      </td>
      <td>
        <a href="https://www.tensorflow.org/datasets/catalog/mnist">
          MNIST
        </a>
      </td>
    </tr>
    <!-- Experiment -->
    <tr>
      <td>
        <img src="assets/images/rock_paper_scissors.png" alt="Rock Paper Scissors" width="150" />
      </td>
      <td>
        <b>Rock Paper Scissors (CNN)</b>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/RockPaperScissorsCNN">
          <img src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&message=Demo&color=green" alt="Launch demo">
        </a>
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <code>Convolutional&nbsp;Neural&nbsp;Network</code>,
        <code>CNN</code>,
        <code>Tensorflow</code>,
        <code>Keras</code>
      </td>
      <td>
        <a href="http://www.laurencemoroney.com/rock-paper-scissors-dataset/">
          RPS
        </a>
      </td>
    </tr>
    <!-- Experiment -->
    <tr>
      <td>
        <img src="assets/images/objects_detection_ssdlite_mobilenet_v2.jpg" alt="Objects detection" width="150" />
      </td>
      <td>
        <b>Objects Detection (MobileNetV2)</b>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/ObjectsDetectionSSDLiteMobilenetV2">
          <img src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&message=Demo&color=green" alt="Launch demo">
        </a>
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <code>Convolutional&nbsp;Neural&nbsp;Network</code>,
        <code>MobileNetV2</code>,
        <code>SSDLite</code>,
        <code>CNN</code>,
        <code>Tensorflow</code>,
      </td>
      <td>
        <a href="http://cocodataset.org/#home">
          COCO
        </a>
      </td>
    </tr>
    <!-- Experiment -->
    <tr>
      <td>
        <img src="assets/images/image_classification_mobilenet_v2.png" alt="Objects detection" width="150" />
      </td>
      <td>
        <b>Image Classification (MobileNetV2)</b>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/ImageClassificationMobilenetV2">
          <img src="https://img.shields.io/static/v1?label=%F0%9F%8E%A8%20Launch&message=Demo&color=green" alt="Launch demo">
        </a>
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <code>Convolutional&nbsp;Neural&nbsp;Network</code>,
        <code>MobileNetV2</code>,
        <code>CNN</code>,
        <code>Tensorflow</code>,
      </td>
      <td>
        <a href="http://image-net.org/explore">
          ImageNet
        </a>
      </td>
    </tr>
  </tbody>
</table>

### Recurrent Neural Networks (RNN)

<table>
  <thead>
    <tr>
      <th align="left" width="150" style="width: 150px !important"> </th>
      <th align="left" width="200" style="width: 200px !important">Experiment</th>
      <th align="left" width="140" style="width: 140px !important">Model demo & training</th>
      <th align="left">Tags</th>
      <th align="left">Dataset</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>

## How to use this repository locally

### Setup virtual environment for Experiments

```bash
# Create "experiments" environment (from the project root folder).
python3 -m venv .virtualenvs/experiments

# Activate environment.
source .virtualenvs/experiments/bin/activate
# or if you use Fish...
source .virtualenvs/experiments/bin/activate.fish
```

To quit an environment run `deactivate`.

### Install dependencies

```bash
# Upgrade pip and setuptools to the latest versions.
pip install --upgrade pip setuptools

# Install packages
pip install -r requirements.txt
```

To install new packages run `pip install package-name`. To add new packages to the requirements run `pip freeze > requirements.txt`.

### Launch Jupyter locally

```bash
# Launch Jupyter server.
jupyter notebook
```

Jupyter will be available locally at `http://localhost:8888/`. Experiments notebooks may be found in `experiments` folder.

### Launch demos locally

```bash
# Switch to demos folder from project root.
cd demos

# Install all dependencies.
yarn install

# Start demo server on http. 
yarn start

# Or start demo server on https (for camera access to work locally).
yarn start-https
```

Demos will be available locally at `http://localhost:3000/` or at `https://localhost:3000/`.

### Convert models

The `converter` environment is used to convert the models that were trained during the experiments from `.h5` Keras format to Javascript understandable formats (`.json` and `.bin`) for further usage in Demo application.

```bash
# Create "converter" environment (from the project root folder).
python3 -m venv .virtualenvs/converter

# Activate "converter" environment.
source .virtualenvs/converter/bin/activate
# or if you use Fish...
source .virtualenvs/converter/bin/activate.fish

# Install converter requirements.
pip install -r requirements.converter.txt
```

The conversion of `.h5` model to `.json` and `.bin` formats is done by using [tfjs-converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter):

For example:

```bash
tensorflowjs_converter --input_format keras \
  ./experiments/digits_recognition_mlp/digits_recognition_mlp.h5 \
  ./demos/public/models/digits_recognition_mlp
```

### Requirements

Recommended versions:

- Python: `> 3.7.3`.
- Node: `>= 12.4.0`.
- Yarn: `>= 1.13.0`.

In case if you have Python version `3.7.3` you might experience `RuntimeError: dictionary changed size during iteration` error when trying to `import tensorflow` (see the [issue](https://github.com/tensorflow/tensorflow/issues/33183)).
