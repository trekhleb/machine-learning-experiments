import React, {useState, useEffect} from 'react';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import { makeStyles } from '@material-ui/core/styles';
import LinearProgress from '@material-ui/core/LinearProgress';
import * as tf from '@tensorflow/tfjs';
import DeleteIcon from '@material-ui/icons/Delete';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';

import Canvas from '../../shared/Canvas';
import type { CanvasImages } from '../../shared/Canvas';
import { MODELS_PATH } from '../../../constants/links';
import type { Experiment } from '../types';
import cover from './cover.png';

const experimentSlug = 'DigitsRecognition';
const experimentName = 'Digits Recognition (MLP)';
const experimentDescription = 'Hand-written digits recognition using MLP (Multilayer Perceptron)';

const canvasWidth = 200;
const canvasHeight = 200;

const modelPath = `${MODELS_PATH}/digits_recognition/model.json`;

const useStyles = makeStyles(() => ({
  paper: {
    width: canvasWidth,
    height: canvasHeight
  },
  recognizedDigit: {
    height: '100%',
    fontSize: '10rem',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
}));

const DigitsRecognition = (): Node => {
  const classes = useStyles();

  const [recognizedDigit, setRecognizedDigit] = useState(null);
  const [digitImageData, setDigitImageData] = useState(null);
  const [model, setModel] = useState(null);
  const [modelLoadingProgress, setModelLoadingProgress] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  const onProgress = (progress) => {
    setModelLoadingProgress(progress);
  };

  useEffect(() => {
    tf.loadLayersModel(modelPath, {onProgress})
      .then((model) => {
        setModel(model);
      })
      .catch((e) => {
        console.error(e);
      })
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
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight])
      .mean(colorsAxis)
      .toFloat()
      .reshape([1, modelInputWidth, modelInputHeight])
      .div(255);

    const prediction = model.predict(tensor);
    const recognizedDigit = prediction.argMax(1).dataSync()[0];
    setRecognizedDigit(recognizedDigit);
  };

  if (modelLoadingProgress < 1) {
    return (
      <Box>
        Loading the model
        <LinearProgress variant="determinate" value={modelLoadingProgress * 100} />
      </Box>
    );
  }

  return (
    <Box display="flex" flexDirection="row">
      <Paper className={classes.paper}>
        <Canvas
          width={canvasWidth}
          height={canvasHeight}
          onDrawEnd={onDrawEnd}
        />  
      </Paper>

      <Box display="flex" flexDirection="column" alignItems="flex-start">
        <Button
          variant="outlined"
          onClick={onRecognize}
          startIcon={<PlayArrowIcon />}
          disabled={!digitImageData}
        >
          Recognize
        </Button>

        <Button
          variant="outlined"
          onClick={onClearCanvas}
          startIcon={<DeleteIcon />}
          disabled={!digitImageData}
        >
          Clear
        </Button>
      </Box>

      <Paper className={classes.paper}>
        <Box className={classes.recognizedDigit}>
          {recognizedDigit}
        </Box>
      </Paper>
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: DigitsRecognition,
  cover,
};

export default experiment;
