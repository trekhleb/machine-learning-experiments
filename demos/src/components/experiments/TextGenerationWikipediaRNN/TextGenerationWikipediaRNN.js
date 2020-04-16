// @flow
import React from 'react';
import type { Node } from 'react';

import type { Experiment } from '../types';
import cover from '../../../images/text_generation_wikipedia_rnn.jpg';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import TextGenerator from '../../shared/TextGenerator';
import modelVocabulary from './vocabulary';

const experimentSlug = 'TextGenerationWikipediaRNN';
const experimentName = 'Wikipedia Text Generation (RNN)';
const experimentDescription = 'Write like Shakespeare. Generate a text using Recurrent Neural Network (RNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/text_generation_wikipedia_rnn/text_generation_wikipedia_rnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/text_generation_shakespeare_rnn/model.json`;

const TextGenerationWikipediaRNN = (): Node => {
  const description = 'Start writing (like Shakespeare) and RNN will continue writing (like Shakespeare) by generating the rest of the text for you.';
  return (
    <TextGenerator
      modelPath={modelPath}
      modelVocabulary={modelVocabulary}
      description={description}
    />
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: TextGenerationWikipediaRNN,
  notebookUrl,
  cover,
};

export default experiment;
