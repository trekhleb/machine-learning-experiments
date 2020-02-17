# 🤖 Machine Learning Experiments

Collection of interactive machine-learning experiments.

_☝ Remember the notebooks and demos in this repository are just **experiments** or playgrounds and **not** a production ready, fine-tuned, super-performant re-usable code_

**How to use this repo**

**@TODO:** Finish this list

- See demo (how to apply models)
- See notebooks (how to train models)
- Checkout and play around locally
- Paste charts from https://arxiv.org/pdf/1611.10012.pdf (+ net configurations?)
- Comparison of CNN computer vision algorithms (Speed, accuracy, etc)
- Machine learning cycle diagram (Train, Deploy, Evaluate)
- Add a link to Homemade ML repo (See how to implement ML algorithms without frameworks)

## Demo

@TODO: Add demo link and instructions: http://trekhleb.github.io/machine-learning-experiments

## Experiments

<table>
  <thead>
    <tr>
      <th align="left" width="150" style="width: 150px !important"> </th>
      <th align="left">Experiment</th>
      <th align="left" width="140" style="width: 140px !important">Model training</th>
      <th align="left">Model usage</th>
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
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/DigitsRecognitionMLP">
          ▶️&nbsp;Launch&nbsp;demo
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
    <!-- Experiment -->
    <tr>
      <td>
        <img src="assets/images/digits_recognition_cnn.png" alt="Handwritten digits recognition (CNN)" />
      </td>
      <td>
        <b>Handwritten digits recognition (CNN)</b>
      </td>
      <td>
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/DigitsRecognitionCNN">
          ▶️&nbsp;Launch&nbsp;demo
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
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/RockPaperScissorsCNN">
          ▶️&nbsp;Launch&nbsp;demo
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
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/ObjectsDetectionSSDLiteMobilenetV2">
          ▶️&nbsp;Launch&nbsp;demo
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
        <a href="https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in Binder"/>
        </a>
        <a href="https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
        </a>
      </td>
      <td>
        <a href="https://trekhleb.github.io/machine-learning-experiments/experiments/ImageClassificationMobilenetV2">
          ▶️&nbsp;Launch&nbsp;demo
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

## How to use this repository

### Virtual environment for Experiments

This environment is used by default to run Jupyter notebooks with experiments.

#### Create environment for Experiments

```bash
python3 -m venv .virtualenvs/experiments
```

#### Activate environment for Experiments

For `shell`:

```bash
source .virtualenvs/experiments/bin/activate
```

For `fish`:

```bash
source .virtualenvs/experiments/bin/activate.fish
```

### Quitting virtual environments

```bash
deactivate
```

### Upgrading `pip` and `setuptools`

```bash
pip install --upgrade pip setuptools
```

### Add new package (optional)

```bash
pip install package
```

### Save added package to `requirements.txt`

```bash
pip freeze > requirements.txt
```

### Install packages

```bash
pip install -r requirements.txt
```

### Launching Jupyter

```bash
jupyter notebook
```

### Launching demos locally

```bash
cd demos
yarn install
yarn start
```

## Converting Keras models into web-format for Demo app

The `converter` environment is used to convert the models that were trained during the experiments from `.h5` format to Javascript understandable formats (`.json` and `.bin`) for further usage in Demo application.

### Create environment for Converter

```bash
python3 -m venv .virtualenvs/converter
```

### Activate environment for Converter

For `shell`:

```bash
source .virtualenvs/converter/bin/activate
```

For `fish`:

```bash
source .virtualenvs/converter/bin/activate.fish
```

### Install packages

To install packages in `converter` environment run:

```bash
pip install -r requirements.converter.txt
```

### Save added package to `requirements.requirements.txt`

To list installed packages for converter environment you should launch:

```bash
pip freeze > requirements.converter.txt
```

### Converting the models

To convert `.h5` model to `.json` and `.bin` formats for further usage in JavaScript Demos you should use [tfjs-converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter):

```bash
tensorflowjs_converter --input_format keras path/to/my_model.h5 path/to/tfjs_target_dir
```

For example:

```bash
tensorflowjs_converter --input_format keras \
  ./experiments/digits_recognition_mlp/digits_recognition_mlp.h5 \
  ./demos/public/models/digits_recognition_mlp
```

## Known Issues

[Issue](https://github.com/tensorflow/tensorflow/issues/33183)

Recommended Python versions: `> 3.7.3`.

In case if you have Python version `3.7.3` you might experience the following error when trying to `import tensorflow`:

```
RuntimeError: dictionary changed size during iteration
```
