// @flow
/* eslint-disable react/no-array-index-key */
import React, { useState } from 'react';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import * as tf from '@tensorflow/tfjs';
import FormControl from '@material-ui/core/FormControl';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import LinearProgress from '@material-ui/core/LinearProgress';

import type { Experiment } from '../types';
import useLayersModel from '../../../hooks/useLayersModel';
import Snack from '../../shared/Snack';
import cover from '../../../images/numbers_summation_rnn.png';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import modelVocabulary from './vocabulary';
import OneHotBars from '../../shared/OneHotBars';
import type { DataRecord } from '../../shared/OneHotBars';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.NumbersSummationRNN;
const experimentName = 'Numbers Summation (RNN)';
const experimentDescription = 'Treat summation expression of two numbers as characters sequence and let RNN sum them up';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/numbers_summation_rnn/numbers_summation_rnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/numbers_summation_rnn/model.json`;

// We need to add a space paddings to input string.
// This is needed in order to have a same size inputs regardless
// of what kind of numbers user has used (i.e. '11+22  ', '1+1    ').
const maxModelInputLength = 7;

const NumbersSummationRNN = (): Node => {
  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });

  const [inputText, setInputText] = useState('');
  const [inputTextError, setInputTextError] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedText, setGeneratedText] = useState('');
  const [predictionOneHots, setPredictionOneHots] = useState(null);

  const inputFormat = /^[1-9][0-9]?\+[1-9][0-9]?$/;

  const onInputTextChange = (event) => {
    const inputString = event.target.value.trim();
    setInputTextError(!inputFormat.test(inputString));
    setInputText(inputString);
    setGeneratedText('');
    setPredictionOneHots(null);
  };

  const onGenerate = (event) => {
    event.preventDefault();

    if (!model) {
      return;
    }

    if (!inputFormat.test(inputText)) {
      return;
    }

    setIsGenerating(true);
    setGeneratedText('');

    const paddings = new Array(maxModelInputLength - inputText.length).fill(' ').join();
    const inputWithPaddings = inputText + paddings;
    const inputOneHots = Array.from(inputWithPaddings)
      // Convert chars to indices.
      .map(
        (inputChar: string) => modelVocabulary
          .findIndex((vocabChar: string) => vocabChar === inputChar),
      )
      // Filter out not found chars.
      .filter((inputCharIndex: number) => inputCharIndex >= 0)
      // Convert indices to one-hot vectors.
      .map((inputCharIndex: number) => {
        const oneHot: number[] = new Array(modelVocabulary.length).fill(0);
        oneHot[inputCharIndex] = 1;
        return oneHot;
      });

    // Here batch size == 1.
    const batchAxis = 0;
    const inputTextTensor = tf.expandDims(inputOneHots, batchAxis);

    model.resetStates();

    let prediction = model.predict(inputTextTensor);
    prediction = tf.squeeze(prediction, batchAxis);

    const predictionText = tf.argMax(prediction, 1)
      .arraySync()
      .filter((charIndex) => {
        const spaceIndex = modelVocabulary.findIndex((char) => char === ' ');
        return spaceIndex !== charIndex;
      })
      .map((charIndex) => modelVocabulary[charIndex])
      .join('');

    setIsGenerating(false);
    setGeneratedText(predictionText);
    setPredictionOneHots(prediction.arraySync());
  };

  const generatedTextSpinner = isGenerating ? (
    <LinearProgress />
  ) : null;

  const generatedTextElement = generatedText ? (
    <Box mb={1}>
      <Typography variant="h2" component="h2">
        {`${inputText} = ${generatedText}`}
      </Typography>
    </Box>
  ) : null;

  if (!model) {
    if (modelErrorMessage) {
      return <Snack severity="error" message={modelErrorMessage} />;
    }

    return (
      <Box>
        <Box>
          Loading the model
        </Box>
        <LinearProgress />
      </Box>
    );
  }

  let oneHots = null;
  if (predictionOneHots && predictionOneHots.length) {
    const bars = predictionOneHots.map((predictionOneHot, predictionIndex) => {
      const data: DataRecord[] = modelVocabulary.map((label, labelIndex) => ({
        value: predictionOneHot[labelIndex],
        label: label === ' ' ? '␣' : label,
      }));
      const title = `Char #${predictionIndex + 1} prediction`;
      return (
        <Grid item xs={12} sm={4} key={predictionIndex}>
          <Box mb={1}>{title}</Box>
          <OneHotBars data={data} />
        </Grid>
      );
    });
    oneHots = (
      <Box mt={5}>
        <Grid
          container
          spacing={3}
          alignItems="flex-start"
          justify="flex-start"
        >
          {bars}
        </Grid>
      </Box>
    );
  }

  return (
    <>
      <Box mb={3}>
        <ol>
          <li>
            Type summation expression (i.e. 74+83 or 3+98).
          </li>
          <li>
            Press Calculate button.
          </li>
          <li>
            RNN will consume your input one by one as a sequence of characters
            (i.e. 7 → 4 → + → 8 → ...).
          </li>
          <li>
            RNN will output another sequence of characters that should look
            like a summation result (i.e. 1 → 5 → 7).
          </li>
        </ol>
      </Box>
      <form onSubmit={onGenerate}>
        <Grid
          container
          spacing={3}
          alignItems="flex-start"
          justify="flex-start"
        >
          <Grid item xs={12} sm={8}>
            <FormControl fullWidth>
              <TextField
                label="Summation expression"
                value={inputText}
                onChange={onInputTextChange}
                variant="outlined"
                size="small"
                helperText="Max 2 digits per number, i.e. 34+27"
                error={inputTextError}
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
                disabled={isGenerating || inputTextError || !inputText}
              >
                Calculate
              </Button>
            </FormControl>
          </Grid>
        </Grid>
      </form>
      <Box mt={3}>
        {generatedTextSpinner}
        {generatedTextElement}
      </Box>
      {oneHots}
      <Snack severity="error" message={modelErrorMessage} />
    </>
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
