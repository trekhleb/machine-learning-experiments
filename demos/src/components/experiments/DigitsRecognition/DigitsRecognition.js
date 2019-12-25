import React, { useState, useEffect } from 'react';
import Paper from '@material-ui/core/Paper';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import { makeStyles } from '@material-ui/core/styles';
import LinearProgress from '@material-ui/core/LinearProgress';
import DeleteIcon from '@material-ui/icons/Delete';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import * as tf from '@tensorflow/tfjs';

import Canvas from '../../shared/Canvas';
import type { CanvasImages } from '../../shared/Canvas';
import OneHotBars, { valueKey, labelKey } from '../../shared/OneHotBars';
import { MODELS_PATH } from '../../../constants/links';
import type { Experiment } from '../types';
import cover from './cover.png';

const experimentSlug = 'DigitsRecognition';
const experimentName = 'Digits Recognition (MLP)';
const experimentDescription = 'Hand-written digits recognition using Multilayer Perceptron (MLP)';
// @TODO: Add URL to see how the model was trained.
const experimentTrainingURL = 'https://jupyter.com';

const canvasWidth = 200;
const canvasHeight = 200;

const modelPath = `${MODELS_PATH}/digits_recognition/model.json`;

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

const DigitsRecognition = () => {
  const classes = useStyles();

  const [model, setModel] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [recognizedDigit, setRecognizedDigit] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [digitImageData, setDigitImageData] = useState(null);
  const [canvasRevision, setCanvasRevision] = useState(0);

  useEffect(() => {
    tf.loadLayersModel(modelPath)
      .then((layersModel) => {
        setModel(layersModel);
      })
      .catch(() => {
        // @TODO: Display an error in a snackbar.
        setModelError(modelError);
      });
  }, []);

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
    if (!digitImageData) {
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

    const prediction = model.predict(tensor.reshape([1, modelInputWidth, modelInputHeight]));
    const digit = prediction.argMax(1).dataSync()[0];
    setRecognizedDigit(digit);
    setProbabilities(prediction.arraySync()[0].map((probability, index) => ({
      [valueKey]: probability,
      [labelKey]: index,
    })));
  };

  if (!model && !modelError) {
    return (
      <Box>
        <Box>
          Loading the model
        </Box>
        <LinearProgress />
      </Box>
    );
  }

  const oneHotBars = probabilities ? (
    <Box width={200}>
      <Box mb={1}>
        Probabilities
      </Box>
      <OneHotBars data={probabilities} />
    </Box>
  ) : null;

  return (
    <Box>
      <Box display="flex" flexDirection="row" mb={1}>
        <Box mb={3}>
          <Box fontWeight="fontWeightLight" mb={1}>
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
        </Box>

        <Box
          display="flex"
          flexDirection="column"
          alignItems="flex-start"
          justifyContent="center"
          pl={2}
          pr={2}
          mb={1}
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

        <Box mb={1}>
          <Box fontWeight="fontWeightLight" mb={1} whiteSpace="nowrap">
            Recognized digit will appear here
          </Box>
          <Paper className={classes.paper}>
            <Box className={classes.recognizedDigit}>
              {recognizedDigit}
            </Box>
          </Paper>
        </Box>

        {oneHotBars}
      </Box>

      <Box mb={1}>
        This model has a disadvantage that the digit should be big and centered.
        If you would try to draw the small digit and in the corner the recognition
        will most probably fail. To overcome this limitation the CNN might be used.
      </Box>
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: DigitsRecognition,
  trainingURL: experimentTrainingURL,
  cover,
};

export default experiment;
