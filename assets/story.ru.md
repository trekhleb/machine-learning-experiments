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

1. 🗓 Поскольку **Python** мог быть хорошим выбором для того, чтобы начать экспериментировать с машинным обучением я решил изучить его базовый синтаксис. В результате появился проект [🐍 Playground and Cheatsheet for Learning Python](https://github.com/trekhleb/learn-python). Он был создан с одной стороны для того, чтобы практиковаться в написании кода на Python, а также в качестве "шпаргалки" с базовым синтаксисом, чтобы в нужный момент можно было быстро подсмотреть вещи наподобие `dict_via_comprehension = {x: x**2 for x in (2, 4, 6)}`.

2. 🗓 После ознакомления с Python-ом я хотел чуть больше углубиться в **математическую** часть машинного обучения. В итоге после прохождения замечательного [курса](https://www.coursera.org/learn/machine-learning) от Andrew Ng на Coursera я создал проект [🤖 Homemade Machine Learning](https://github.com/trekhleb/homemade-machine-learning). Это была очередная "шпаргалка для себя-же" с базовыми алгоритмами машинного обучения, такими как линейная регрессия, логистическая регрессия, алгоритм k-средних, многослойный перцептрон (или персептрон? 🤔) и прочие.

3. 🗓 Моей следующей попыткой "поиграться в машинное обучение" стал [🤖 NanoNeuron](https://github.com/trekhleb/nano-neuron). Это были 7 простых JavaScript функций, которые должны были дать понимание читателю о том, как же машина все-таки может "учиться".

4. 🗓 После окончания очередного прекрасного [курса по Deep Learning](https://www.coursera.org/specializations/deep-learning) от все того-же Andrew Ng на Coursera я решил больше попрактиковаться с **многослойными перцептронами** (multilayer perceptrons), **сверточными** и **рекуррентными нейронными сетями** (convolutional and recurrent neural networks). На этот раз, вместо того, чтобы реализовывать их с нуля я решил воспользоваться уже готовым фреймворком. В итоге я начал с [TensorFlow 2](https://www.tensorflow.org/) с поддержкой [Keras](https://www.tensorflow.org/guide/keras/overview). Я так же не хотел фокусироваться на математике (позволив фреймворку сделать свое дело), вместо этого хотелось написать что-то более практичное и интерактивное, что-то, что можно было бы протестировать прямо в браузере телефона. В результате появился новый проект [🤖 Interactive Machine Learning Experiments](https://github.com/trekhleb/machine-learning-experiments), на котором я и хочу остановиться более детально в этой статье.

## Технологический стек

### Тренировка моделей

- 🏋🏻‍ Для тренировки моделей я использовал [Keras](https://www.tensorflow.org/guide/keras/overview), как часть [TensorFlow 2](https://www.tensorflow.org/). Поскольку до этого у меня не было опыта с фреймворками для машинного обучения мне нужно было с какого-то из них начать. Один из ключевых факторов, который мне понравился в TensorFlow было наличие сразу двух его версий: версии на Python и [версии на JavaScript](https://www.tensorflow.org/js), у которых был схожий API. В итоге я использовал Python версия для тренировки, а JavaScript версию библиотека для демо-приложения. 

- 🏋🏻‍ Я тренировал модели на Python внутри [Jupyter](https://jupyter.org/) ноутбуков локально. Иногда использовал [Colab](https://colab.research.google.com/), чтобы воспользоваться GPU и тем самым ускорить тренировку.

- 💻 Большинство моделей были натренированы на CPU старого доброго MacBook Pro (2,9 GHz Dual-Core Intel Core i5).

- 🔢 И конечно же было никак не обойтись без [NumPy](https://numpy.org/) для матричных (тензорных) операций.   

### Демонстрация моделей

- 🏋🏻‍ Я использовал [TensorFlow.js](https://www.tensorflow.org/js) для того, чтобы воспользоваться в браузере заранее натренированными на предыдущем шаге (в Jupyter ноутбуке) моделями.

- ♻️ Для конвертирования моделей из формата _HDF5_ в формат _TensorFlow.js Layers_ я использовал [TensorFlow.js converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter). Это конечно же может быть неэффективно загружать всю модель в браузер целиком (речь ведь идет о мегабайтах данных) вместо того, чтобы делать предсказания вызывая модель удаленно через HTTP запросы, но, снова-таки, вспомним, что речь идет об _экспериментах_, а не о зрелой и оптимизированной архитектуре, которую можно брать и сразу же использовать для "продакшна". С точки зрения простоты подхода я также хотел избежать развертывания отдельного сервера с HTTP API для предсказаний моделей.

- 👨🏻‍🎨 [Демонстрационное приложение](http://trekhleb.github.io/machine-learning-experiments) было создано на [React](https://reactjs.org/) с использованием [create-react-app](https://github.com/facebook/create-react-app) стартера с поддержкой [Flow](https://flow.org/en/) по умолчания для проверки типов.

- 💅🏻 Для стайлинга я воспользовался библиотекой [Material UI](https://material-ui.com/). Я хотел, как говориться, "убить двух зайцев сразу" и заодно попробовать новый для себя фреймворк для пользовательских интерфейсов (прости, [Bootstrap](https://getbootstrap.com/) 🤷🏻‍). 

## Эксперименты

Демо-страничка с экспериментами, а также Jupyter ноутбуки с деталями тренировки доступны по следующим ссылкам:

- 🎨 [**Запустить Демо с машинными экспериментами**](http://trekhleb.github.io/machine-learning-experiments)
- 🏋️ [**Просмотреть детали тренировки каждой из моделей**](https://github.com/trekhleb/machine-learning-experiments)

### Эксперименты с многослойным перцептроном (Multilayer Perceptron, MLP)

#### Распознавание цифр

Вы рисуете цифру, а модель пытается ее распознать.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/DigitsRecognitionMLP)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_mlp/digits_recognition_mlp.ipynb)

![Handwritten Digits Recognition](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/03-digits-recognition.gif)

#### Распознавание эскизов

Вы рисуете эскиз, а модель пытается его распознать.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/SketchRecognitionMLP)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_mlp/sketch_recognition_mlp.ipynb)

![Handwritten Sketch Recognition](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/04-sketch-recognition.gif)

### Эксперименты со сверточными нейронными сетями (Convolutional Neural Network, CNN)

#### Распознавание цифр (CNN)

Вы рисуете цифру, а модель пытается ее распознать. Этот эксперимент похож на предыдущий из раздела MLP, но на этот раз модель использует CNN.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/DigitsRecognitionCNN)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb)

![Handwritten Digits Recognition (CNN)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/03-digits-recognition.gif)

#### Распознавание эскизов (CNN)

Вы рисуете эскиз, а модель пытается его распознать. Этот эксперимент похож на предыдущий из раздела MLP, но на этот раз модель использует CNN.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/SketchRecognitionCNN)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/sketch_recognition_cnn/sketch_recognition_cnn.ipynb)

![Handwritten Sketch Recognition (CNN)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/04-sketch-recognition.gif)

#### Камень-Ножницы-Бумага (CNN)

Вы играете в камень-ножницы-бумагу с моделью. Этот эксперимент использует CNN, натренированную с нуля.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/RockPaperScissorsCNN)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb)

![Rock Paper Scissors (CNN)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/05-rock-paper-scissors.gif)

#### Rock Paper Scissors (MobilenetV2)

Вы играете в камень-ножницы-бумагу с моделью. Эта модель использует трансферное обучение, основанное на сети [MobilenetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/RockPaperScissorsMobilenetV2)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb)

![Rock Paper Scissors (MobilenetV2)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/06-rock-paper-scissors-mobilenet.gif)

#### Распознавание объектов (MobileNetV2)

Вы показываете модели ваше окружение (используя камеру ноутбука или телефона), а модель пытается определить предметы на видео и распознать их. Эта модель использует сеть [MobilenetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/ObjectsDetectionSSDLiteMobilenetV2)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb)

![Objects Detection (MobileNetV2)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/07-objects-detection.gif)

#### Классификация изображений (MobileNetV2)

Вы загружаете изображение, а модель пытается его классифицировать в зависимости от того, что она "видит" на картинке. Модель использует сеть [MobilenetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/ImageClassificationMobilenetV2)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb)

![Image Classification (MobileNetV2)](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/08-image-classification.gif)

### Эксперименты с рекуррентными нейронными сетями (Recurrent Neural Networks, RNN) 

#### Суммирование чисел

Вы набираете выражение (например, `17+38`) и модель предсказывает результат (например `55`). Интересность этой модели заключается в том, что она воспринимает выражение на входе, как _последовательность_ символов (как текст). Модель учится "переводить" последовательность `1` → `17` → `17+` → `17+3` → `17+38` на входе в другую текстовую последовательность `55` на выходе. Считайте, что модель скорее делает перевод испанского `Hola` в английское `Hello`, чем оперирует математическими сущностями.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/NumbersSummationRNN)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb)

![Numbers Summation](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/09-numbers-summation.gif)

#### Генерация текста в стиле Шекспира

Вы начинаете поэму как Шекспир, а модель пытается ее продолжить как Шекспир. Ключевое слово "пытается" 😀.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/TextGenerationShakespeareRNN)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb)

![Shakespeare Text Generation](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/10-shakespeare-text-generation.gif)

#### Генерация текста в стиле Wikipedia

Вы начинаете печатать Wiki статью, а модель продолжает.

- 🎨 [Демо](https://trekhleb.github.io/machine-learning-experiments/#/experiments/TextGenerationWikipediaRNN)
- 🏋️ [Тренировка модели в Jupyter](https://nbviewer.jupyter.org/v2/gh/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb)
- ️🏋️  [Тренировка модели в Colab](https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb)

![Wikipedia Text Generation](https://raw.githubusercontent.com/trekhleb/machine-learning-experiments/master/assets/images/story/11-wikipedia-text-generation.gif)

## Планы

Как я упомянул выше, главная задача [репозитория](https://github.com/trekhleb/machine-learning-experiments) - быть тренировочной площадкой, песочницей для обучения машинному обучению (привет, каламбур 🙌🏻). Поэтому в планах - **продолжать учиться и экспериментировать** с различными задачами в области Deep Learning. Такими интересным задачами могут быть:

- Определение эмоций
- Перенос стиля
- Машинный перевод
- Генерация изображения (например тех-же написанных цифр)
- и пр.

Другой интересной возможностью является **более тонкая настройка уже имеющихся моделей**, чтобы сделать их более точными. Мне кажется это может дать более глубокое понимание того, как преодолевать переученность/недоученность моделей и как поступать в случаях, когда точность модели застревает на `60%` для тренировочных и валидационных данных и не хочет больше улучшаться 🤔.

В любом случае, я надеюсь, что вы найдете что-то полезное и интересное в [репозитории](https://github.com/trekhleb/machine-learning-experiments) в контексте обучения машинных моделей, ну или по крайней мере обыграете компьютер в камень-ножницы-бумагу 😀 

Успешного обучения! 🤖
