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

import Snack from './Snack';

type SequenceLength = 100 | 200 | 400 | 800;
type Unexpectedness = 0.1 | 0.5 | 1 | 1.5;

const defaultSequenceLengthValue: SequenceLength = 400;
const defaultUnexpectednessValue: Unexpectedness = 0.1;

const sequenceLengths: SequenceLength[] = [100, 200, 400, 800];
const unexpectednessList: Unexpectedness[] = [0.1, 0.5, 1, 1.5];

const defaultProps = {
  maxInputLength: 100,
  defaultSequenceLength: defaultSequenceLengthValue,
  defaultUnexpectedness: defaultUnexpectednessValue,
};

type TextGeneratorProps = {
  modelPath: string,
  modelVocabulary: string[],
  maxInputLength?: number,
  defaultSequenceLength?: SequenceLength,
  defaultUnexpectedness?: Unexpectedness,
};

const TextGenerator = (props: TextGeneratorProps): Node => {
  const {
    modelPath,
    modelVocabulary,
    maxInputLength,
    defaultSequenceLength = defaultSequenceLengthValue,
    defaultUnexpectedness = defaultUnexpectednessValue,
  } = props;

  const [model, setModel] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
  const [inputText, setInputText] = useState('');
  const [sequenceLength, setSequenceLength] = useState(defaultSequenceLength);
  const [unexpectedness, setUnexpectedness] = useState(defaultUnexpectedness);
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
  }, [modelPath, setErrorMessage, setModel]);

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

    const inputTextIndices = Array.from(inputText)
      .map(
        (inputChar: string) => modelVocabulary
          .findIndex((vocabChar: string) => vocabChar === inputChar),
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
              {sequenceLengths.map(
                (seqLength) => (
                  <MenuItem key={seqLength} value={seqLength}>
                    {seqLength}
                  </MenuItem>
                ),
              )}
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
              {unexpectednessList.map(
                (randomness) => (
                  <MenuItem key={randomness} value={randomness}>
                    {randomness}
                  </MenuItem>
                ),
              )}
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
              disabled={isGenerating}
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

TextGenerator.defaultProps = defaultProps;

export default TextGenerator;
