// @flow
import React from 'react';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';

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
  const description = '';

  const stopSign = '␣';
  const stopWordTitle = '📗 ';
  const stopWordIngredients = '\n🥕\n\n';
  const stopWordInstructions = '\n📝\n\n';

  const preProcessOutput = (generatedText: string): string => {
    let preProcessedText = generatedText;
    preProcessedText = preProcessedText.replace(new RegExp(stopSign, 'g'), '');
    preProcessedText = preProcessedText.replace(new RegExp(stopWordTitle), '📗 [NAME]\n\n');
    preProcessedText = preProcessedText.replace(new RegExp(stopWordInstructions), '\n📝 [INSTRUCTIONS]\n\n');
    preProcessedText = preProcessedText.replace(new RegExp(stopWordIngredients), '\n🥕 [INGREDIENTS]\n\n');
    return preProcessedText;
  };

  return (
    <Box>
      <p>
        Let Recurrent Neural Network generate a randomly weird recipe for you.
      </p>

      <Box mb={4}>
        <ul>
          <li>
            ⚠️ This is just for fun and not for real cooking
          </li>
          <li>
            ℹ️ It is ok to leave start text blank (you may experiment and try something like <code>Mushroom</code> though)
          </li>
          <li>
            ℹ️ Generator is case-sensitive
          </li>
          <li>
            💡 If recipe looks like a garbage, try to generate it one more time or change
            the parameters, the result might become more meaningful afterwards
          </li>
          <li>
            🥑 If still no luck, check <a href="https://www.instagram.com/home_full_of_recipes/">real recipes here</a>
          </li>
        </ul>
      </Box>

      <TextGenerator
        modelPath={modelPath}
        modelVocabulary={modelVocabulary}
        preProcessOutput={preProcessOutput}
        description={description}
        defaultSequenceLength={800}
        defaultUnexpectedness={0.4}
        sequencePrefix={stopWordTitle}
        inputRequired={false}
        inputDisabled={false}
      />
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: RecipeGenerationRNN,
  notebookUrl,
  cover,
  inputTextExamples: ['Mushroom', 'Sweet', 'O', ''],
};

export default experiment;
