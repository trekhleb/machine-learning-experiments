// @flow
import React from 'react';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';

import type { Experiment } from '../types';
import cover from '../../../images/clothes_generation_dcgan.jpg';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.ClothesGenerationDCGAN;
const experimentName = 'Clothes Generation (DCGAN)';
const experimentDescription = 'Generate clothes out of random noise using Deep Convolution Generative Adversarial Network (DCGAN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/clothes_generation_dcgan/clothes_generation_dcgan.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/clothes_generation_dcgan/model.json`;

const ClothesGenerationDCGAN = (): Node => {
  return (
    <Box>
      <Box>
        A
        {' '}
        <b>Generative Adversarial Network</b>
        {' '}
        (GAN) is a class of machine learning frameworks.
        Two models (generator and discriminator) are trained simultaneously.
      </Box>
      <ul>
        <li>
          <span role="img" aria-label="GENERATOR">🦹🏻</span>‍
          {' '}
          GENERATOR (the artist) learns to create images that look real.
        </li>
        <li>
          <span role="img" aria-label="DISCRIMINATOR">👮🏻</span>
          {' '}
          DISCRIMINATOR (the art critic) learns to distinguish real images apart from fakes.
        </li>
      </ul>
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: ClothesGenerationDCGAN,
  notebookUrl,
  cover,
};

export default experiment;
