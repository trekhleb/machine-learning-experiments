# 🤖 Интерактивные эксперименты с машинным обучением

Привет, читатели!

## Если вкратце

Я создал новый проект [**🤖 Интерактивные эксперименты с машинным обучением**](https://github.com/trekhleb/machine-learning-experiments) на GitHub. Каждый эксперимент состоит из 🏋️ _Jupyter/Colab ноутбука_, показывающего как модель тренировалась, и 🎨 _Демо странички_, показывающей модель в действии прямо в вашем браузере. 

Несмотря на то, что машинные модели в репозитории могут быть немного "туповатенькими" (помните, это всего-лишь эксперименты, а не вылизанный код, готовый к "заливке на продакшн" и дальнейшему управлению новыми Tesla), они будут стараться как могут чтобы:

- 🖌 Распознать цифры и прочие эскизы, которые вы нарисуете в браузере
- 📸 Определить и распознать объекты на видео из вашей камеры
- 🌅 Классифицировать изображения, загруженные вами
- 📝 Написать с вами поэму в стиле Шекспира
- ✊🖐✌️ И даже поиграть с вами в камень-ножницы-бумагу
- и пр. 

Я тренировал модели на _Python_ с использованием _TensorFlow 2_ с поддержкой _Keras_. Для демо-приложения я использовал _React_ и _JavaScript_ версию _Tensorflow_. 

![Интерактивные эксперименты с машинным обучением](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/repository-cover.png)

## Производительность моделей

⚠️ Для начала, давайте определимся с нашими ожиданиями.️ Репозиторий содержит **эксперименты** с машинным обучением, а **не** готовые к "заливке на продакшн", оптимизированные и тонко настроенные модели. Этот проект скорее похож на песочницу, в которой можно учиться и тренироваться работе с алгоритмами машинного обучения и разными наборами данных. Обученные модели могут быть недостаточно точными (например, иметь 60% точности вместо ожидаемых, пускай, 97%), а также могу быть переученными и недоученными (overfitting vs underfitting).

Поэтому иногда вы можете увидеть что-то вроде:

![Тупенькая машинная модель](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/01-dumb-model.png)

Но будьте терпеливы, иногда эта же модель может выдавать что-то более "умное" 🤓:

![Более умная машинная модель](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/02-smart-model.png)

## Предыстория

Я [инженер-программист](https://www.linkedin.com/in/trekhleb/) и последние несколько лет занимаюсь в основном full-stack программированием (веб-проекты). В свободное от работы время, в качестве хобби, я решил углубиться в тему машинного обучения, чтобы в первую очередь для себя-же сделать эту тему менее _магической_ и более _математической_.

1. 🗓 Поскольку **Python** мог быть хорошим выбором для того, чтобы начать экспериментировать с машинным обучением я решил выучить его базовый синтаксис. В результате появился проект [🐍 Playground and Cheatsheet for Learning Python](https://github.com/trekhleb/learn-python). Он был создан с одной стороны для того, чтобы практиковаться в написании кода на Python, а так же в качестве "шпаргалки" с базовым синтаксисом, чтобы в нужный момент можно было быстро подсмотреть вещи на подобии `dict_via_comprehension = {x: x**2 for x in (2, 4, 6)}`.

2. 🗓 После ознакомления с Python-ом я хотел чуть больше углубиться в **математическую** часть машинного обучения. В итоге после прохождения замечательного [курса](https://www.coursera.org/learn/machine-learning) от Andrew Ng на Coursera я создал проект [🤖 Homemade Machine Learning](https://github.com/trekhleb/homemade-machine-learning). Это была очередная "шпаргалка для себя-же" с базовыми алгоритмами машинного обучения, такими как линейная регрессия, логистическая регрессия, алгоритм k-средних, многослойный перцептрон (или персептрон? 🤔) и прочие.

3. 🗓 Моей следующей попыткой "поиграться в машинное обучение" стал [🤖 NanoNeuron](https://github.com/trekhleb/nano-neuron). Это были 7 простых JavaScript функций, которые должны были дать понимание читателю о том, как же машина все-таки может "учиться".

4. 🗓 После окончания очередного прекрасного [курса по Deep Learning](https://www.coursera.org/specializations/deep-learning) от все того-же Andrew Ng на Coursera я решил больше попрактиковаться с **многослойными перцептронами** (multilayer perceptrons), **сверточными** и **рекурентными нейронными сетями** (convolutional and recurrent neural networks). На этот раз, вместо того, чтобы реализовывать их с нуля я решил воспользоваться уже готовым фреймворком. В итоге я начал с [TensorFlow 2](https://www.tensorflow.org/) с поддержкой [Keras](https://www.tensorflow.org/guide/keras/overview). Я так же не хотел фокусироваться на математике (позволив фреймворку сделать свое дело), вместо этого хотелось написать что-то более практичное и интерактивное, что-то, что можно было бы протестировать прямо в браузере телефона. В результате появился новый проект [🤖 Interactive Machine Learning Experiments](https://github.com/trekhleb/machine-learning-experiments), на котором я и хочу остановиться более детально в этой статье.

## Технологический стек

### Тренировка моделей

- 🏋🏻‍ Для тренировки моделей я использовал [Keras](https://www.tensorflow.org/guide/keras/overview), как часть [TensorFlow 2](https://www.tensorflow.org/). Поскольку до этого у меня не было опыта с фреймворками для машинного обучения мне нужно было с какого-то из них начать. Один из ключевых факторов, который мне понравился в TensorFlow было наличие сразу двух его версий: версии на Python и [версии на JavaScript](https://www.tensorflow.org/js), у которых был схожий API. В итоге я использовал Python версия для тренировки, а JavaScript версию библиотека для демо-приложения. 

- 🏋🏻‍ Я тренировал модели на Python внутри [Jupyter](https://jupyter.org/) ноутбуков локально. Иногда использовал [Colab](https://colab.research.google.com/), чтобы воспользоваться GPU и тем самым ускорить тренировку.

- 💻 Большинство моделей были натренированны на CPU старого доброго MacBook Pro (2,9 GHz Dual-Core Intel Core i5).

- 🔢 И конечно же было никак не обойтись без [NumPy](https://numpy.org/) для матричных (тензорных) операций.   

### Демонстрация моделей

- 🏋🏻‍ Я использовал [TensorFlow.js](https://www.tensorflow.org/js) для того, чтобы воспользоваться в браузере заранее натренированными на предыдущем шаге (в Jupyter ноутбуке) моделями.

- ♻️ Для конвертирования моделей из формата _HDF5_ в формат _TensorFlow.js Layers_ я использовал [TensorFlow.js converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter). Это конечно же может быть неэффективно загружать всю модель в браузер целиком (речь ведь идет о мегабайтах данных) вместо того, чтобы делать предсказания вызывая модель удаленно через HTTP запросы, но, снова-таки, вспомним, что речь идет об _экспериментах_, а не о зрелой и оптимизированной архитектуре, которую можно брать и сразу же использовать для "продакшна". С точки зрения простоты подхода я так же хотел избежать развертывания отдельного сервера с HTTP API дле предсказаний моделей.

- 👨🏻‍🎨 [Демонстрационное приложение](http://trekhleb.github.io/machine-learning-experiments) было создано на [React](https://reactjs.org/) с использованием [create-react-app](https://github.com/facebook/create-react-app) стартера с поддержкой [Flow](https://flow.org/en/) по умолчания для проверки типов.

- 💅🏻 Для стайлинга я воспользовался библиотекой [Material UI](https://material-ui.com/). Я хотел, как говориться, "убить двух зайцев сразу" и заодно попробовать новый для себя фреймворк для пользовательских интерфейсов (прости, [Bootstrap](https://getbootstrap.com/) 🤷🏻‍). 

## Эксперименты

Демо-страничка с экспериментами, а так же Jupyter ноутбуки с деталями тренировки доступны по следующим ссылкам:

- 🎨 [**Запустить Демо с машинными экспериментами**](http://trekhleb.github.io/machine-learning-experiments)
- 🏋️ [**Просмотреть детали тренировки каждой из моделей**](https://github.com/trekhleb/machine-learning-experiments)

### Experiments with Multilayer Perceptron (MLP)

> A [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a class of feedforward artificial neural network (ANN). Multilayer perceptrons are sometimes referred to as "vanilla" neural networks (composed of multiple layers of perceptrons), especially when they have a single hidden layer.

#### Handwritten Digits Recognition

You draw a digit, and the model tries to recognize it.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/DigitsRecognitionMLP)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb)

![Handwritten Digits Recognition](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/03-digits-recognition.gif)

#### Handwritten Sketch Recognition

You draw a sketch, and the model tries to recognize it.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/SketchRecognitionMLP)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb)

![Handwritten Sketch Recognition](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/04-sketch-recognition.gif)

### Experiments with Convolutional Neural Networks (CNN)

> A [convolutional neural network (CNN, or ConvNet)](https://en.wikipedia.org/wiki/Convolutional_neural_network) is a class of deep neural networks, most commonly applied to analyzing visual imagery (photos, videos). They are used for detecting and classifying objects on photos and videos, style transfer, face recognition, pose estimation etc.

#### Handwritten Digits Recognition (CNN)

You draw a digit, and the model tries to recognize it. This experiment is similar to the one from MLP section, but it uses CNN under the hood.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/DigitsRecognitionCNN)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb)

![Handwritten Digits Recognition (CNN)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/03-digits-recognition.gif)

#### Handwritten Sketch Recognition (CNN)

You draw a sketch, and the model tries to recognize it. This experiment is similar to the one from MLP section, but it uses CNN under the hood.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/SketchRecognitionCNN)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb)

![Handwritten Sketch Recognition (CNN)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/04-sketch-recognition.gif)

#### Rock Paper Scissors (CNN)

You play a Rock-Paper-Scissors game with the model. This experiment uses CNN that is trained from scratch.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/RockPaperScissorsCNN)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb)

![Rock Paper Scissors (CNN)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/05-rock-paper-scissors.gif)

#### Rock Paper Scissors (MobilenetV2)

You play a Rock-Paper-Scissors game with the model. This model uses transfer learning and is based on [MobilenetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/RockPaperScissorsMobilenetV2)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb)

![Rock Paper Scissors (MobilenetV2)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/06-rock-paper-scissors-mobilenet.gif)

#### Objects Detection (MobileNetV2)

You show to the model your environment through your camera, and it will try to detect and recognize the objects. This model uses transfer learning and is based on [MobilenetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/ObjectsDetectionSSDLiteMobilenetV2)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb)

![Objects Detection (MobileNetV2)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/07-objects-detection.gif)

#### Image Classification (MobileNetV2)

You upload a picture, and the model tries to classify it depending on what it "sees" on the picture. This model uses transfer learning and is based on [MobilenetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/ImageClassificationMobilenetV2)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb)

![Image Classification (MobileNetV2)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/08-image-classification.gif)

### Experiments with Recurrent Neural Networks (RNN) 

> A [recurrent neural network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) is a class of deep neural networks, most commonly applied to sequence-based data like speech, voice, text or music. They are used for machine translation, speech recognition, voice synthesis etc.

#### Numbers Summation

You type a summation expression (i.e. `17+38`), and the model predicts the result (i.e. `55`). The interesting part here is that the model treats the input as a _sequence_, meaning it learned that when you type a sequence `1` → `17` → `17+` → `17+3` → `17+38` it "translates" it to another sequence `55`. You may think about it as translating a Spanish `Hola` sequence to English `Hello`.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/NumbersSummationRNN)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb)

![Numbers Summation](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/09-numbers-summation.gif)

#### Shakespeare Text Generation

You start typing a poem like Shakespeare, and the model will continue it like Shakespeare. At least it will try to do so 😀.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/TextGenerationShakespeareRNN)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb)

![Shakespeare Text Generation](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/10-shakespeare-text-generation.gif)

#### Wikipedia Text Generation

You start typing a Wiki article, and the model tries to continue it.

- 🎨 [Demo](https://trekhleb.github.io/machine-learning-experiments/#/experiments/TextGenerationWikipediaRNN)
- 🏋️ [Training in Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb)
- ️🏋️  [Training in Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb)

![Wikipedia Text Generation](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/11-wikipedia-text-generation.gif)

## Future plans

As I've mentioned above the main purpose of [the repository](https://github.com/trekhleb/machine-learning-experiments) is to be more like a playground for learning rather than for production-ready models. Therefore, the main plan is to **continue learning and experimenting** with deep-learning challenges and approaches. The next interesting challenges to play with might be:

- Emotions detection
- Style transfer
- Language translation
- Generating images (i.e. handwritten numbers)
- etc.

Another interesting opportunity would be to **tune existing models to make them more performant**. I believe it might give a better understanding of how to overcome overfitting and underfitting and what to do with the model if it just stuck on `60%` accuracy level for both training and validation sets and doesn't want to improve anymore 🤔.

Anyways, I hope you might find some useful insights for models training from [the repository](https://github.com/trekhleb/machine-learning-experiments) or at least to have some fun playing around with the demos!

Happy learning! 🤖
