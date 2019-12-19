# 🤖 Machine Learning Experiments

Collection of interactive machine-learning experiments.

## Demo

@TODO: Add demo link and instructions

## Experiments

- Digits recognition

| Gif preview | Name | Complexity | How to use the model | How to train the model | Topics | Technology |
| ----------- | ---- | ---------- | -------------------- | ---------------------- | ------ | ---------- |
| Gif image | Handwritten digits recognition | ⭐️⭐️ | demo 🏄‍♂️| 🏋️‍♂️ | CNN, NN | Tensorflow |

## How to use this repository

### Virtual environment for Experiments

This environment is used to run Jupyter notebooks with experiments. This environment is used by default for further documentation. If some commands need to be run in another environment (i.e. in `convertor` environment) this will be specified explicitly.

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

### Virtual environment for Model Convertor

This environment is used to convert the models that were trained during the experiments from `.h5` format to Javascript understandable formats (`.json` and `.bin`) for further usage in Demo application.

#### Create environment for Converter

```bash
python3 -m venv .virtualenvs/convertor
```

#### Activate environment for Convertor

For `shell`:

```bash
source .virtualenvs/convertor/bin/activate
```

For `fish`:

```bash
source .virtualenvs/convertor/bin/activate.fish
```

### Quitting virtual environments

```bash
deactivate
```

### Upgrading `pip`

```bash
pip install --upgrade pip
```

### Add new package (optional)

```bash
pip install package
```

### Save added package to `requirements.txt`

```bash
pip freeze > requirements.txt
```

To list installed packages for convertor environment you should launch:

```bash
pip freeze > requirements.convertor.txt
```

### Install packages

```bash
pip install -r requirements.txt
```

To install packages in `convertor` environment run:

```bash
pip install -r requirements.convertor.txt
```

### Launching Jupyter

```bash
jupyter notebook
```

### Converting the models

To convert `.h5` model to `.json` and `.bin` formats for further usage in JavaScript Demos you should run:

```bash
tensorflowjs_converter --input_format keras path/to/my_model.h5 path/to/tfjs_target_dir
```

For example:

```bash
tensorflowjs_converter --input_format keras ./experiments/digits_recognition/digits_recognition.h5 ./demos/public/models/digits_recognition
```

### Launching demos locally

```bash
cd demos
yarn install
yarn start
```
