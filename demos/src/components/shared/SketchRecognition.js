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
import Typography from '@material-ui/core/Typography';
// $FlowFixMe
import * as tf from '@tensorflow/tfjs';

import Canvas from './Canvas';
import type { CanvasImages } from './Canvas';

import useLayersModel from '../../hooks/useLayersModel';

const additionalGuessesNum = 3;

const canvasWidth = 200;
const canvasHeight = 200;

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

type SketchRecognitionProps = {
  modelPath: string,
  labels: string[],
};

const SketchRecognition = (props: SketchRecognitionProps): Node => {
  const { modelPath, labels } = props;

  const classes = useStyles();

  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });
  const [recognizedCategoryIndex, setRecognizedCategoryIndex] = useState(null);
  const [guessIndices, setGuessIndices] = useState(null);
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
    setGuessIndices(null);
    setSketchImageData(null);
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

    const probabilities = prediction
      .arraySync()[0]
      .map((probability, index) => ({ probability, index }))
      .sort((probabilityA, probabilityB) => (probabilityB.probability - probabilityA.probability))
      .map((probability) => probability.index)
      .slice(1, additionalGuessesNum + 1);

    setGuessIndices(probabilities);
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
          lineWidth={8}
        />
      </Paper>
    </>
  );

  // eslint-disable-next-line max-len
  const recognizedCategory = recognizedCategoryIndex !== null && recognizedCategoryIndex < labels.length
    ? labels[recognizedCategoryIndex]
    : null;

  const additionalGuesses = guessIndices ? guessIndices.map((guessId, guessIndex) => (
    <React.Fragment key={guessId}>
      <Grid item>
        or
      </Grid>
      <Grid item>
        <Typography variant={`h${3 + guessIndex}`} component={`h${3 + guessIndex}`}>
          {labels[guessId]}
        </Typography>
      </Grid>
    </React.Fragment>
  )) : null;

  const recognizedCategoryElement = recognizedCategory ? (
    <Box mt={2}>
      <Grid container spacing={2} alignItems="center" justify="flex-start">
        <Grid item>
          It looks like
        </Grid>
        <Grid item>
          <Typography variant="h1" component="h1">
            {recognizedCategory}
          </Typography>
        </Grid>
        {additionalGuesses}
      </Grid>
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

  return (
    <Box>
      <Box mb={3}>
        Draw a sketch and pre-trained model will try to recognize
        what it is among 345 sketch categories. Model is trained on
        <a href="https://quickdraw.withgoogle.com/data">QuickDraw dataset</a>
        .
      </Box>

      <Grid container spacing={3} alignItems="center" justify="flex-start">
        <Grid item>
          {canvasPaper}
        </Grid>

        <Grid item>
          {buttons}
        </Grid>
      </Grid>

      {recognizedCategoryElement}
    </Box>
  );
};

export default SketchRecognition;
