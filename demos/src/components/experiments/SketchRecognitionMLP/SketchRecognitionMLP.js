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
// $FlowFixMe
import * as tf from '@tensorflow/tfjs';

import Canvas from '../../shared/Canvas';
import type { CanvasImages } from '../../shared/Canvas';
import OneHotBars, { valueKey, labelKey } from '../../shared/OneHotBars';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';

import labels from './labels';

import cover from '../../../images/sketch_recognition_mlp.png';
import inputImageExample0 from './input-examples/0.png';
import inputImageExample1 from './input-examples/1.png';
import inputImageExample2 from './input-examples/2.png';
import inputImageExample3 from './input-examples/3.png';
import inputImageExample4 from './input-examples/4.png';
import useLayersModel from '../../../hooks/useLayersModel';
import Typography from '@material-ui/core/Typography';

const experimentSlug = 'SketchRecognitionMLP';
const experimentName = 'Sketch Recognition (MLP)';
const experimentDescription = 'Hand-written sketch recognition using Multilayer Perceptron (MLP)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/sketch_recognition_mlp/sketch_recognition_mlp.ipynb`;
const inputImagesExamples = [
  inputImageExample0,
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
  inputImageExample4,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/sketch_recognition_mlp/model.json`;

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

const SketchRecognitionMLP = (): Node => {
  const classes = useStyles();

  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });
  const [recognizedCategoryIndex, setRecognizedCategoryIndex] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [sketchImageData, setSketchImageData] = useState(null);
  const [canvasRevision, setCanvasRevision] = useState(0);

  const onDrawEnd = (canvasImages: CanvasImages) => {
    if (!canvasImages.imageData) {
      return;
    }
    setSketchImageData(canvasImages.imageData);
  };

  const onClearCanvas = () => {
    setRecognizedCategoryIndex(null);
    setSketchImageData(null);
    setProbabilities(null);
    setCanvasRevision(canvasRevision + 1);
  };

  const onRecognize = () => {
    if (!sketchImageData || !model) {
      return;
    }

    const modelInputWidth = model.layers[0].input.shape[1];
    const modelInputHeight = model.layers[0].input.shape[2];
    const colorsAxis = 2;

    const tensor = tf.browser
      .fromPixels(sketchImageData)
      // Resize image to fit neural network input.
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight])
      // Calculate grey-scale average across channels.
      .mean(colorsAxis)
      // Invert image colors to fit neural network model input.
      .mul(-1)
      .add(255)
      // Normalize.
      .div(255)
      // Reshape.
      .reshape([1, modelInputWidth, modelInputHeight, 1]);

    const prediction = model.predict(tensor);
    const categoryIndex = prediction.argMax(1).dataSync()[0];
    setRecognizedCategoryIndex(categoryIndex);
    // setProbabilities(prediction.arraySync()[0].map((probability, index) => ({
    //   [valueKey]: Math.floor(10 * probability) / 10,
    //   [labelKey]: index,
    // })));
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
        sketch figure here
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

  const recognizedCategory =
    recognizedCategoryIndex !== null && recognizedCategoryIndex < labels.length
      ? labels[recognizedCategoryIndex]
      : null;

  const recognizedCategoryElement = recognizedCategory ? (
    <Box>
      <Typography variant="h2" component="h2">
        {recognizedCategory}
      </Typography>
    </Box>
  ) : null;

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
          disabled={!sketchImageData}
        >
          Recognize
        </Button>
      </Box>

      <Box mb={1}>
        <Button
          color="secondary"
          onClick={onClearCanvas}
          startIcon={<DeleteIcon />}
          disabled={!sketchImageData}
        >
          Clear
        </Button>
      </Box>
    </Box>
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
        {recognizedCategoryElement}
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
  component: SketchRecognitionMLP,
  notebookUrl,
  cover,
  inputImageExamples: {
    images: inputImagesExamples,
    imageWidth: 50,
  },
};

export default experiment;
