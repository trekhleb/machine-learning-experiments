// @flow
import React, { useState } from 'react';
import type { Node } from 'react';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import FormControl from '@material-ui/core/FormControl';
import Box from '@material-ui/core/Box';
import LinearProgress from '@material-ui/core/LinearProgress';
import { makeStyles } from '@material-ui/core/styles';

import type { Experiment } from '../types';
import cover from '../../../images/text_generation_shakespeare_rnn.jpg';
import { ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL } from '../../../constants/links';

const experimentSlug = 'TextGenerationShakespeareRNN';
const experimentName = 'Shakespeare Text Generation (RNN)';
const experimentDescription = 'Write like Shakespeare. Generate a text using Recurrent Neural Network (RNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb`;

const useStyles = makeStyles(() => ({
  header: {
    display: 'flex',
    fontSize: '20px',
    fontWeight: 400,
    marginBottom: '10px',
  },
}));

const TextGenerationShakespeareRNN = (): Node => {
  const maxInputLength = 100;

  const [inputText, setInputText] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const classes = useStyles();

  const onInputTextChange = (event) => {
    setInputText(event.target.value);
  };

  const onGenerate = (event) => {
    event.preventDefault();
    setIsGenerating(true);
  };

  const generatedTextHeader = generatedText ? (
    <Box className={classes.header}>
      Generated text
    </Box>
  ) : null;

  const generatedTextSpinner = isGenerating ? (
    <LinearProgress />
  ) : null;

  const generatedTextElement = generatedText ? (
    <Box>
      {generatedText}
    </Box>
  ) : null;

  return (
    <form onSubmit={onGenerate}>
      <Grid
        container
        spacing={3}
        alignItems="center"
        justify="flex-start"
      >
        <Grid item xs={12} sm={8}>
          <FormControl fullWidth>
            <TextField
              label="Type the beginning of the text"
              value={inputText}
              onChange={onInputTextChange}
              variant="outlined"
              inputProps={{
                maxLength: maxInputLength,
              }}
              required
            />
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth>
            <Button
              variant="contained"
              color="primary"
              size="large"
              type="submit"
            >
              Generate Text
            </Button>
          </FormControl>
        </Grid>
      </Grid>

      <Box mt={3}>
        {generatedTextHeader}
        {generatedTextSpinner}
        {generatedTextElement}
      </Box>
    </form>
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
