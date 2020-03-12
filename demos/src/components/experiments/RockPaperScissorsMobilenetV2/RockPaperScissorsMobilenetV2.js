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

import cover from './cover.png';
import inputImageExample1 from './input-examples/rock.png';
import inputImageExample2 from './input-examples/paper.png';
import inputImageExample3 from './input-examples/scissors.png';
import RockPaperScissors from '../../shared/RockPaperScissors';

const experimentSlug = 'RockPaperScissorsMobilenetV2';
const experimentName = 'Rock Paper Scissors (MobilenetV2)';
const experimentDescription = 'Play Rock Paper Scissors game against computer using Convolutional Neural Network (MobilenetV2)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/rock_paper_scissors_mobilenet_v2/rock_paper_scissors_mobilenet_v2.ipynb`;
const inputImagesExamples = [
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/rock_paper_scissors_mobilenet_v2/model.json`;

/* eslint-disable react/jsx-one-expression-per-line */
const RockPaperScissorsMobilenetV2 = (): Node => {
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
  component: RockPaperScissorsMobilenetV2,
  notebookUrl,
  cover,
  inputImageExamples: {
    imageWidth: 150,
    images: inputImagesExamples,
  },
};

export default experiment;
