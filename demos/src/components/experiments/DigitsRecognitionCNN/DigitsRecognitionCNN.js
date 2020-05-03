// @flow
import React, { useState } from 'react';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import LinearProgress from '@material-ui/core/LinearProgress';
import DeleteIcon from '@material-ui/icons/Delete';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import * as tf from '@tensorflow/tfjs';

import Canvas from '../../shared/Canvas';
import type { CanvasImages } from '../../shared/Canvas';
import OneHotBars, { valueKey, labelKey } from '../../shared/OneHotBars';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';

import cover from '../../../images/digits_recognition_cnn.png';
import inputImageExample0 from './input-examples/0.png';
import inputImageExample1 from './input-examples/1.png';
import inputImageExample2 from './input-examples/2.png';
import inputImageExample3 from './input-examples/3.png';
import inputImageExample4 from './input-examples/4.png';
import inputImageExample9 from './input-examples/9.png';
import useLayersModel from '../../../hooks/useLayersModel';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.DigitsRecognitionCNN;
const experimentName = 'Digits Recognition (CNN)';
const experimentDescription = 'Hand-written digits recognition using Convolutional Neural Network (CNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/digits_recognition_cnn/digits_recognition_cnn.ipynb`;
const inputImagesExamples = [
  inputImageExample0,
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
  inputImageExample4,
  inputImageExample9,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/digits_recognition_cnn/model.json`;

const canvasWidth = 200;
const canvasHeight = 200;
const oneHotBarWidth = 200;
const oneHotBarHeight = 90;

const useStyles = makeStyles(() => ({
  paper: {
    width: canvasWidth,
    height: canvasHeight,
    overflow: 'hidden',
  },
  recognizedDigit: {
    height: '100%',
    fontSize: '10rem',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
}));

const DigitsRecognitionCNN = (): Node => {
  const classes = useStyles();

  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });
  const [recognizedDigit, setRecognizedDigit] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [digitImageData, setDigitImageData] = useState(null);
  const [canvasRevision, setCanvasRevision] = useState(0);

  const onDrawEnd = (canvasImages: CanvasImages) => {
    if (!canvasImages.imageData) {
      return;
    }
    setDigitImageData(canvasImages.imageData);
  };

  const onClearCanvas = () => {
    setRecognizedDigit(null);
    setDigitImageData(null);
    setProbabilities(null);
    setCanvasRevision(canvasRevision + 1);
  };

  const onRecognize = () => {
    if (!digitImageData || !model) {
      return;
    }

    const modelInputWidth = model.layers[0].input.shape[1];
    const modelInputHeight = model.layers[0].input.shape[2];
    const colorsAxis = 2;

    const tensor = tf.browser
      .fromPixels(digitImageData)
      // Resize image to fit neural network input.
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight])
      // Calculate grey-scale average across channels.
      .mean(colorsAxis)
      // Invert image colors to fit neural network model input.
      .mul(-1)
      .add(255)
      // Normalize.
      .div(255);

    const prediction = model.predict(
      // Reshape and add one dimension for the pixel color to match CNN input size
      tensor.reshape([1, modelInputWidth, modelInputHeight, 1]),
    );
    const digit = prediction.argMax(1).dataSync()[0];
    setRecognizedDigit(digit);
    setProbabilities(prediction.arraySync()[0].map((probability, index) => ({
      [valueKey]: Math.floor(10 * probability) / 10,
      [labelKey]: index,
    })));
  };

  if (!model && !modelErrorMessage) {
    return (
      <Box>
        <Box>
          Loading the model
        </Box>
        <LinearProgress />
      </Box>
    );
  }

  const canvasPaper = (
    <>
      <Box mb={1}>
        Draw
        {' '}
        <b>one BIG</b>
        {' '}
        digit here
      </Box>
      <Paper className={classes.paper}>
        <Canvas
          width={canvasWidth}
          height={canvasHeight}
          onDrawEnd={onDrawEnd}
          revision={canvasRevision}
        />
      </Paper>
    </>
  );

  const buttons = (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="flex-start"
      justifyContent="center"
    >
      <Box mb={1}>
        <Button
          color="primary"
          onClick={onRecognize}
          startIcon={<PlayArrowIcon />}
          disabled={!digitImageData}
        >
          Recognize
        </Button>
      </Box>

      <Box mb={1}>
        <Button
          color="secondary"
          onClick={onClearCanvas}
          startIcon={<DeleteIcon />}
          disabled={!digitImageData}
        >
          Clear
        </Button>
      </Box>
    </Box>
  );

  const digitsPaper = (
    <>
      <Box mb={1} whiteSpace="nowrap">
        Recognized digit appears here
      </Box>
      <Paper className={classes.paper}>
        <Box className={classes.recognizedDigit}>
          {recognizedDigit}
        </Box>
      </Paper>
    </>
  );

  const oneHotBars = probabilities ? (
    <Box width={oneHotBarWidth}>
      <Box mb={1}>
        Probabilities
      </Box>
      <OneHotBars data={probabilities} height={oneHotBarHeight} />
    </Box>
  ) : null;

  return (
    <Grid container spacing={3} alignItems="center" justify="flex-start">
      <Grid item>
        {canvasPaper}
      </Grid>

      <Grid item>
        {buttons}
      </Grid>

      <Grid item>
        {digitsPaper}
      </Grid>

      <Grid item>
        {oneHotBars}
      </Grid>
    </Grid>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: DigitsRecognitionCNN,
  notebookUrl,
  cover,
  inputImageExamples: {
    images: inputImagesExamples,
    imageWidth: 50,
  },
  similarExperiments: [
    experimentsSlugs.DigitsRecognitionMLP,
  ],
};

export default experiment;
