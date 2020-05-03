// @flow
import React from 'react';
import type { Node } from 'react';

import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';

import labels from './labels';

import cover from '../../../images/sketch_recognition_cnn.png';
import inputImageExample0 from './input-examples/0.png';
import inputImageExample1 from './input-examples/1.png';
import inputImageExample2 from './input-examples/2.png';
import inputImageExample3 from './input-examples/3.png';
import inputImageExample4 from './input-examples/4.png';
import inputImageExample5 from './input-examples/5.png';
import inputImageExample6 from './input-examples/6.png';
import inputImageExample7 from './input-examples/7.png';
import inputImageExample8 from './input-examples/8.png';
import inputImageExample9 from './input-examples/9.png';
import inputImageExample10 from './input-examples/10.png';
import inputImageExample11 from './input-examples/11.png';
import inputImageExample12 from './input-examples/12.png';
import SketchRecognition from '../../shared/SketchRecognition';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.SketchRecognitionCNN;
const experimentName = 'Sketch Recognition (CNN)';
const experimentDescription = 'Hand-written sketch recognition using Convolutional Neural Network (CNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/sketch_recognition_cnn/sketch_recognition_cnn.ipynb`;
const inputImagesExamples = [
  inputImageExample0,
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
  inputImageExample4,
  inputImageExample5,
  inputImageExample6,
  inputImageExample7,
  inputImageExample8,
  inputImageExample9,
  inputImageExample10,
  inputImageExample11,
  inputImageExample12,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/sketch_recognition_cnn/model.json`;

const SketchRecognitionCNN = (): Node => (
  <SketchRecognition
    labels={labels}
    modelPath={modelPath}
  />
);

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: SketchRecognitionCNN,
  notebookUrl,
  cover,
  inputImageExamples: {
    images: inputImagesExamples,
    imageWidth: 50,
  },
  similarExperiments: [
    experimentsSlugs.SketchRecognitionMLP,
  ],
};

export default experiment;
