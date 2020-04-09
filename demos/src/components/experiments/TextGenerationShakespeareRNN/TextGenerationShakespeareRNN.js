// @flow
import React, { useCallback, useEffect, useState } from 'react';
import type { Node } from 'react';
import * as tf from '@tensorflow/tfjs';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import FormControl from '@material-ui/core/FormControl';
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';
import LinearProgress from '@material-ui/core/LinearProgress';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';

import type { Experiment } from '../types';
import cover from '../../../images/text_generation_shakespeare_rnn.jpg';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import Snack from '../../shared/Snack';

const experimentSlug = 'TextGenerationShakespeareRNN';
const experimentName = 'Shakespeare Text Generation (RNN)';
const experimentDescription = 'Write like Shakespeare. Generate a text using Recurrent Neural Network (RNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/text_generation_shakespeare_rnn/text_generation_shakespeare_rnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/text_generation_shakespeare_rnn/model.json`;

const TextGenerationShakespeareRNN = (): Node => {
  const maxInputLength = 100;

  const [model, setModel] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
  const [inputText, setInputText] = useState('');
  const [sequenceLength, setSequenceLength] = useState(400);
  const [unexpectedness, setUnexpectedness] = useState(0.1);
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const warmupModel = () => {
    if (model && !modelIsWarm) {
      const fakeInput = tf.tensor([[0, 1]]);
      model.predict(fakeInput);
    }
  };

  const warmupModelCallback = useCallback(warmupModel, [model, modelIsWarm]);

  // Effect for loading the model.
  useEffect(() => {
    tf.loadLayersModel(modelPath)
      .then((layersModel) => {
        setModel(layersModel);
      })
      .catch((e) => {
        setErrorMessage(e.message);
      });
  }, [setErrorMessage, setModel]);

  // Effect for warming up a model.
  useEffect(() => {
    if (model && !modelIsWarm) {
      warmupModelCallback();
      setModelIsWarm(true);
    }
  }, [model, modelIsWarm, setModelIsWarm, warmupModelCallback]);

  const onInputTextChange = (event) => {
    setInputText(event.target.value);
  };

  const onGenerate = (event) => {
    event.preventDefault();
    if (!model) {
      return;
    }

    setIsGenerating(true);
    setGeneratedText('');

    const modelVocabulary = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];

    const inputTextIndices = Array.from(inputText)
      .map(
        (inputChar: string) => modelVocabulary
          .findIndex((vocabChar: string) => vocabChar === inputChar)
      )
      .filter((inputCharIndex: number) => inputCharIndex >= 0);

    // Here batch size == 1.
    const batchAxis = 0;
    const inputTextTensor = tf.expandDims(inputTextIndices, batchAxis);

    const textGenerated = [];

    model.resetStates();

    let inputTensor = inputTextTensor;

    for (let charIndex = 0; charIndex < sequenceLength; charIndex += 1) {
      let prediction = model.predict(inputTensor);

      // Remove the batch dimension.
      prediction = tf.squeeze(prediction, batchAxis);

      // Using a categorical distribution to predict the character returned by the model.
      prediction = tf.div(prediction, unexpectedness);
      const predictionArray = prediction.arraySync();
      const lastPrediction = predictionArray[predictionArray.length - 1];
      const nextCharIndex = tf.multinomial(tf.tensor(lastPrediction), 1).arraySync()[0];

      textGenerated.push(modelVocabulary[nextCharIndex]);

      inputTensor = tf.expandDims([nextCharIndex], batchAxis);
    }

    setIsGenerating(false);
    setGeneratedText(`${inputText}${textGenerated.join('')}...`);
  };

  const generatedTextSpinner = isGenerating ? (
    <LinearProgress />
  ) : null;

  const generatedTextElement = generatedText ? (
    <Box>
      <Box mb={1}>
        <Typography variant="h6" component="h3">
          Generated text
        </Typography>
      </Box>
      <Typography variant="body2" component="div">
        <TextField
          value={generatedText}
          variant="outlined"
          multiline
          fullWidth
        />
      </Typography>
    </Box>
  ) : null;

  if (!model) {
    if (errorMessage) {
      return <Snack severity="error" message={errorMessage} />;
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

  if (!modelIsWarm) {
    return (
      <Box>
        <Box>
          Preparing the model...
        </Box>
      </Box>
    );
  }

  return (
    <form onSubmit={onGenerate} className="mt-5">
      <Box mb={2}>
        Start writing (like Shakespeare) and RNN will continue writing (like Shakespeare)
        by generating the rest of the text for you.
      </Box>
      <Grid
        container
        spacing={3}
        alignItems="flex-start"
        justify="flex-start"
      >
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth>
            <TextField
              label="Start the text"
              value={inputText}
              onChange={onInputTextChange}
              variant="outlined"
              size="small"
              helperText="English letters and spaces allowed"
              inputProps={{
                maxLength: maxInputLength,
              }}
              required
            />
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={3}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel id="sequence-length-select-label">
              Text length
            </InputLabel>
            <Select
              labelId="sequence-length-select-label"
              id="sequence-length-select"
              value={sequenceLength}
              onChange={(e) => setSequenceLength(e.target.value)}
              label="Text length"
            >
              <MenuItem value={100}>100</MenuItem>
              <MenuItem value={200}>200</MenuItem>
              <MenuItem value={400}>400</MenuItem>
              <MenuItem value={800}>800</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={2}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel id="unexpectedness-select-label">
              Fuzziness
            </InputLabel>
            <Select
              labelId="unexpectedness-select-label"
              id="unexpectedness-select"
              value={unexpectedness}
              onChange={(e) => setUnexpectedness(e.target.value)}
              label="Fuzziness"
            >
              <MenuItem value={0.1}>0.1</MenuItem>
              <MenuItem value={0.5}>0.5</MenuItem>
              <MenuItem value={1}>1</MenuItem>
              <MenuItem value={1.5}>1.5</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={3}>
          <FormControl fullWidth>
            <Button
              variant="contained"
              color="primary"
              size="large"
              type="submit"
            >
              Generate
            </Button>
          </FormControl>
        </Grid>
      </Grid>

      <Box mt={3}>
        {generatedTextSpinner}
        {generatedTextElement}
      </Box>

      <Snack severity="error" message={errorMessage} />
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
