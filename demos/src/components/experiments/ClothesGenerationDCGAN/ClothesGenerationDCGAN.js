// @flow
import React, { useState } from 'react';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import { makeStyles } from '@material-ui/core/styles';
import * as tf from '@tensorflow/tfjs';
import LinearProgress from '@material-ui/core/LinearProgress';
import Button from '@material-ui/core/Button';
import DeleteIcon from '@material-ui/icons/Delete';
import GestureIcon from '@material-ui/icons/Gesture';
import TextureIcon from '@material-ui/icons/Texture';

import type { Experiment } from '../types';
import cover from '../../../images/clothes_generation_dcgan.jpg';
import demoImage from '../../../images/clothes_generation_dcgan.gif';
import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import { experimentsSlugs } from '../types';
import Canvas from '../../shared/Canvas';
import type { CanvasImages } from '../../shared/Canvas';
import useLayersModel from '../../../hooks/useLayersModel';

const experimentSlug = experimentsSlugs.ClothesGenerationDCGAN;
const experimentName = 'Clothes Generation (DCGAN)';
const experimentDescription = 'Generate clothes out of random noise using Deep Convolution Generative Adversarial Network (DCGAN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/clothes_generation_dcgan/clothes_generation_dcgan.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/clothes_generation_dcgan/model.json`;

const paperWidth = 200;
const paperHeight = 200;

const inputCanvasWidth = paperWidth;
const inputCanvasHeight = paperHeight;

const outputCanvasWidth = 60;
const outputCanvasHeight = 60;

const useStyles = makeStyles(() => ({
  paper: {
    width: paperWidth,
    height: paperHeight,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  canvas: {

  },
}));

const ClothesGenerationDCGAN = (): Node => {
  const classes = useStyles();

  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });

  const [canvasRevision, setCanvasRevision] = useState(0);
  const [noiseImage, setNoiseImage] = useState(null);
  const [inputImage, setInputImage] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);

  const onDrawEnd = (canvasImages: CanvasImages) => {
    if (!canvasImages.imageData) {
      return;
    }
    setNoiseImage(null);
    setInputImage(canvasImages.imageData);
  };

  const onClearCanvas = () => {
    setGeneratedImage(null);
    setInputImage(null);
    setNoiseImage(null);
    setCanvasRevision(canvasRevision + 1);
  };

  const onGenerateFromNoise = () => {
    if (!model) {
      return;
    }

    const colorChannels = 1;
    const noise = tf
      .randomNormal([inputCanvasWidth, inputCanvasHeight, colorChannels])
      .mul(127.5)
      .add(255);
    setNoiseImage(noise.arraySync());

    const modelInputSize = model.layers[0].input.shape[1];
    const normalNoise = tf.randomNormal([1, modelInputSize]);
    const outputTensor = model.predict(normalNoise);
    const outputImage = outputTensor
      // Denormalize.
      .mul(127.5)
      .add(127.5)
      // Invert colors.
      .mul(-1)
      .add(255)
      .arraySync()[0];

    setGeneratedImage(outputImage);
  };

  const onGenerate = () => {
    if (!inputImage || !model) {
      return;
    }

    const modelInputSize = model.layers[0].input.shape[1];
    const modelInputWidth = Math.sqrt(modelInputSize);
    const modelInputHeight = Math.sqrt(modelInputSize);
    const colorsAxis = 2;

    const noiseTensor = tf.browser
      .fromPixels(inputImage)
      // Resize image to fit neural network input.
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight])
      // Calculate grey-scale average across channels.
      .mean(colorsAxis)
      // Invert image colors to fit neural network model input.
      .mul(-1)
      .add(255)
      // Normalize.
      .sub(127.5)
      .div(127.5)
      // Reshape
      .reshape([1, modelInputSize]);

    const outputTensor = model.predict(noiseTensor);
    const outputImage = outputTensor
      // Denormalize.
      .mul(127.5)
      .add(127.5)
      // Invert colors.
      .mul(-1)
      .add(255)
      .arraySync()[0];

    setGeneratedImage(outputImage);
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

  const inputCanvasPaper = (
    <>
      <Box mb={1}>
        Draw input
        {' '}
        <b>noise</b>
        {' '}
        image here
      </Box>
      <Paper className={classes.paper}>
        <Canvas
          width={inputCanvasWidth}
          height={inputCanvasHeight}
          onDrawEnd={onDrawEnd}
          revision={canvasRevision}
          image={noiseImage}
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
          onClick={onGenerateFromNoise}
          startIcon={<TextureIcon />}
          disabled={inputImage}
        >
          Generate From Noise
        </Button>
      </Box>

      <Box mb={1}>
        <Button
          color="primary"
          onClick={onGenerate}
          startIcon={<GestureIcon />}
          disabled={!inputImage || noiseImage}
        >
          Generate From Drawing
        </Button>
      </Box>

      <Box mb={1}>
        <Button
          color="secondary"
          onClick={onClearCanvas}
          startIcon={<DeleteIcon />}
          disabled={!inputImage && !noiseImage}
        >
          Clear
        </Button>
      </Box>
    </Box>
  );

  const outputCanvasPaper = (
    <>
      <Box mb={1} whiteSpace="nowrap">
        <b>Generated</b>
        {' '}
        image appears here
      </Box>
      <Paper className={classes.paper}>
        <Canvas
          width={outputCanvasWidth}
          height={outputCanvasHeight}
          revision={canvasRevision}
          image={generatedImage}
          disabled
        />
      </Paper>
    </>
  );

  return (
    <Box>
      <Box>
        A
        {' '}
        <b>Generative Adversarial Network</b>
        {' '}
        (GAN) is a class of machine learning frameworks where
        two models (generator and discriminator) are trained simultaneously.
      </Box>

      <Box mb={3}>
        <ul>
          <li>
            <span role="img" aria-label="GENERATOR">🦹🏻</span>
            ‍
            {' '}
            GENERATOR (the artist) learns to create images that look real.
          </li>
          <li>
            <span role="img" aria-label="DISCRIMINATOR">👮🏻</span>
            {' '}
            DISCRIMINATOR (the art critic) learns to distinguish real images apart from fakes.
          </li>
        </ul>
      </Box>

      <Grid container spacing={3} alignItems="center" justify="flex-start">
        <Grid item>
          {inputCanvasPaper}
        </Grid>

        <Grid item>
          {buttons}
        </Grid>

        <Grid item>
          {outputCanvasPaper}
        </Grid>
      </Grid>

      <Box mt={4}>
        Here is an example of how this model was trained to generate more and more realistic
        clothes images out of random noise inputs.
      </Box>

      <Box mt={3}>
        <img src={demoImage} alt="DCGAN training process demo" />
      </Box>
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: ClothesGenerationDCGAN,
  notebookUrl,
  cover,
};

export default experiment;
