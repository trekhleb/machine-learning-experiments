// @flow
import React from 'react';
import type { Node } from 'react';

import type { Experiment } from '../types';
import cover from '../../../images/numbers_summation_rnn.png';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';

import modelVocabulary from './vocabulary';

const experimentSlug = 'NumbersSummationRNN';
const experimentName = 'Numbers Summation (RNN)';
const experimentDescription = 'Treat summation expression of two numbers as a string and let RNN sum them up';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/numbers_summation_rnn/numbers_summation_rnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/numbers_summation_rnn/model.json`;

const NumbersSummationRNN = (): Node => {
  return (
    <div>
      Experiment here...
    </div>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: NumbersSummationRNN,
  notebookUrl,
  cover,
  inputTextExamples: ['1+20', '74+83', '3+98'],
};

export default experiment;
