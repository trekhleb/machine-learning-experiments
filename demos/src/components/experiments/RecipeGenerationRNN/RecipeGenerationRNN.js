// @flow
import React from 'react';
import type { Node } from 'react';

import type { Experiment } from '../types';
import cover from '../../../images/recipe_generation_rnn.jpg';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import TextGenerator from '../../shared/TextGenerator';
import modelVocabulary from './vocabulary';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.RecipeGenerationRNN;
const experimentName = 'Recipe Generation (RNN)';
const experimentDescription = 'Generate a recipe with ingredients and cooking instructions using Recurrent Neural Network (RNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/recipe_generation_rnn/recipe_generation_rnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/recipe_generation_rnn/model.json`;

const RecipeGenerationRNN = (): Node => {
  const stopWordTitle = '📗 ';
  const description = 'Let Recurrent Neural Network generate a randomly weird recipe for you. This is just for fun and not for actual cooking.';
  return (
    <TextGenerator
      modelPath={modelPath}
      modelVocabulary={modelVocabulary}
      description={description}
      defaultSequenceLength={800}
      defaultUnexpectedness={0.4}
      sequencePrefix={stopWordTitle}
      inputRequired={false}
      inputDisabled
      modelStrict
    />
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: RecipeGenerationRNN,
  notebookUrl,
  cover,
};

export default experiment;
