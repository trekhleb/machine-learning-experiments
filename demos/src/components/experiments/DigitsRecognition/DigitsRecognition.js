import React, {useState, useEffect} from 'react';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import { makeStyles } from '@material-ui/core/styles';
import * as tf from '@tensorflow/tfjs';

import Canvas from '../../shared/Canvas';
import type { CanvasImages } from '../../shared/Canvas';
import {MODELS_PATH} from '../../../constants/links';
import type { Experiment } from '../types';
import cover from './cover.png';

const experimentSlug = 'DigitsRecognition';
const experimentName = 'Digits Recognition';
const experimentDescription = 'Hand-written digits recognition';

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
  const [digitDataURL, setDigitDataURL] = useState(null);
  const [digitBlob, setDigitBlob] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    tf.loadLayersModel(modelPath)
      .then((model) => {
        setModel(model);
      })
      .catch((e) => {
        console.error('Error while fetching the model:', e);
      })
  }, []);

  const onDrawEnd = (canvasImages: CanvasImages) => {
    setDigitDataURL(canvasImages.dataURL);
    setDigitBlob(canvasImages.blob);
  };

  const onClearCanvas = () => {
    setRecognizedDigit(null);
  };

  const onRecognize = () => {
    setRecognizedDigit(3);
  };

  return (
    <Box flexDirection="row" display="flex">
      <Paper className={classes.paper}>
        <Canvas
          width={canvasWidth}
          height={canvasHeight}
          onDrawEnd={onDrawEnd}
        />  
      </Paper>

      <Box>
        <Button onClick={onRecognize}>Recognize</Button>
        <Button onClick={onClearCanvas}>Clear</Button>
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
