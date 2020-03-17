// @flow
import React, { useState } from 'react';
import type { Node } from 'react';
import TextField from '@material-ui/core/TextField';

import type { Experiment } from '../types';
import cover from '../../../images/text_generation_shakespeare_rnn.jpg';
import { ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL } from '../../../constants/links';

const experimentSlug = 'TextGenerationShakespeareRNN';
const experimentName = 'Shakespeare Text Generation (RNN)';
const experimentDescription = 'Write like Shakespeare. Generate a text using Recurrent Neural Network (RNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb`;

const TextGenerationShakespeareRNN = (): Node => {
  const [inputText, setInputText] = useState('');

  const onInputTextChange = (event) => {
    setInputText(event.target.value);
  };

  return (
    <div>
      <TextField
        id="standard-basic"
        label="Type the beginning"
        value={inputText}
        onChange={onInputTextChange}
      />
    </div>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: TextGenerationShakespeareRNN,
  notebookUrl,
  cover,
};

export default experiment;
