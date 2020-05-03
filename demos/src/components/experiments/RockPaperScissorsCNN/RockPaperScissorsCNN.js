// @flow
import React from 'react';
import type { Node } from 'react';

import { ML_EXPERIMENTS_DEMO_MODELS_PATH, ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL } from '../../../constants/links';
import type { Experiment } from '../types';
import Snack from '../../shared/Snack';

import cover from '../../../images/rock_paper_scissors_cnn.jpg';
import inputImageExample1 from './input-examples/rock.png';
import inputImageExample2 from './input-examples/paper.png';
import inputImageExample3 from './input-examples/scissors.png';
import RockPaperScissors from '../../shared/RockPaperScissors';
import useLayersModel from '../../../hooks/useLayersModel';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.RockPaperScissorsCNN;
const experimentName = 'Rock Paper Scissors (CNN)';
const experimentDescription = 'Play Rock Paper Scissors game against computer using Convolutional Neural Network (CNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb`;
const inputImagesExamples = [
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/rock_paper_scissors_cnn/model.json`;

const RockPaperScissorsCNN = (): Node => {
  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });

  return (
    <>
      <RockPaperScissors model={model} />
      <Snack severity="error" message={modelErrorMessage} />
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
  similarExperiments: [
    experimentsSlugs.RockPaperScissorsMobilenetV2,
  ],
};

export default experiment;
