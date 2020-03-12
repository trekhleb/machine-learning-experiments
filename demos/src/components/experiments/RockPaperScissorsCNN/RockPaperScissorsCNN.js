// @flow
import React, {
  useState,
  useEffect,
} from 'react';
import * as tf from '@tensorflow/tfjs';
import type { Node } from 'react';

import { ML_EXPERIMENTS_DEMO_MODELS_PATH, ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL } from '../../../constants/links';
import type { Experiment } from '../types';
import Snack from '../../shared/Snack';

import cover from './rock_paper_scissors_cnn.jpg';
import inputImageExample1 from './input-examples/rock.png';
import inputImageExample2 from './input-examples/paper.png';
import inputImageExample3 from './input-examples/scissors.png';
import RockPaperScissors from '../../shared/RockPaperScissors';

const experimentSlug = 'RockPaperScissorsCNN';
const experimentName = 'Rock Paper Scissors (CNN)';
const experimentDescription = 'Play Rock Paper Scissors game against computer using Convolutional Neural Network (CNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb`;
const inputImagesExamples = [
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/rock_paper_scissors_cnn/model.json`;

/* eslint-disable react/jsx-one-expression-per-line */
const RockPaperScissorsCNN = (): Node => {
  const [model, setModel] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Effect for loading the model.
  useEffect(() => {
    tf.loadLayersModel(modelPath)
      .then((layersModel) => {
        setModel(layersModel);
      })
      .catch((e) => {
        setErrorMessage(e);
      });
  }, [setErrorMessage, setModel]);

  return (
    <>
      <RockPaperScissors model={model} />
      <Snack severity="error" message={errorMessage} />
    </>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: RockPaperScissorsCNN,
  notebookUrl,
  cover,
  inputImageExamples: {
    imageWidth: 150,
    images: inputImagesExamples,
  },
};

export default experiment;
