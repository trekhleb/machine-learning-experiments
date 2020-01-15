import React, { useState, useEffect, useRef } from 'react';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';
import { makeStyles } from '@material-ui/core/styles';
import LinearProgress from '@material-ui/core/LinearProgress';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import PhoneIphoneIcon from '@material-ui/icons/PhoneIphone';
import Paper from '@material-ui/core/Paper';

import {
  // ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';
import cover from './cover.jpeg';
import CanvasShapes from '../../shared/CanvasShapes';
import CameraStream from '../../shared/CameraStream';
import Snack from '../../shared/Snack';

const experimentSlug = 'ObjectsDetection';
const experimentName = 'Objects Detection';
const experimentDescription = 'Detecting objects on the image or camera';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/digits_recognition_mlp/digits_recognition_mlp.ipynb`;

// const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/digits_recognition_mlp/model.json`;
const defaultCameraStreamSize = 300;
const maxCameraStreamSize = 400;

const useStyles = makeStyles(() => ({
  paper: {
    overflow: 'hidden',
  },
}));

const ObjectsDetection = (): Node => {
  const classes = useStyles();

  const cameraStreamWrapper = useRef(null);

  const [width, setWidth] = useState(defaultCameraStreamSize);
  const [height, setHeight] = useState(defaultCameraStreamSize);
  const [model, setModel] = useState(null);
  const [boxes, setBoxes] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const onVideoFrame = (video: HTMLVideoElement) => {
    if (!model) {
      return;
    }
    model.detect(video)
      .then((detections) => {
        setBoxes(detections.map((detection) => ({
          leftTopX: detection.bbox[0],
          leftTopY: detection.bbox[1],
          width: detection.bbox[2],
          height: detection.bbox[3],
          label: `${detection.class}: ${detection.score.toFixed(2)}`,
        })));
      })
      .catch((e) => {
        setErrorMessage('Detection has failed');
      });
  };

  // Load the model.
  useEffect(() => {
    if (cameraStreamWrapper.current && cameraStreamWrapper.current.offsetWidth) {
      const size = Math.min(maxCameraStreamSize, cameraStreamWrapper.current.offsetWidth);
      setWidth(size);
      setHeight(size);
      console.log({size});
    }

    if (model) {
      return;
    }

    cocoSsd.load()
      .then((cocoSsdModel) => {
        setModel(cocoSsdModel);
      })
      .catch((e) => {
        // @TODO: Display an error in a snackbar.
        setErrorMessage('Model cannot be loaded');
      });
  }, [model, setErrorMessage, setModel]);

  if (!model) {
    return (
      <Box>
        <Box>
          Loading the model
        </Box>
        <LinearProgress />
      </Box>
    );
  }

  const canvasStyle = {
    marginTop: -1 * height,
  };

  return (
    <Box ref={cameraStreamWrapper}>
      <Box mb={1} display="flex" alignItems="center">
        <PhoneIphoneIcon />
        {' '}
        Your camera stream appears here
      </Box>
      <Paper className={classes.paper} style={{ width, height }}>
        <CameraStream
          width={width}
          height={height}
          maxWidth={maxCameraStreamSize}
          maxHeight={maxCameraStreamSize}
          onVideoFrame={onVideoFrame}
        />
      </Paper>
      <Box style={canvasStyle}>
        <CanvasShapes
          canvasWidth={width}
          canvasHeight={height}
          boxes={boxes}
        />
      </Box>
      <Snack severity="error" message={errorMessage} />
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: ObjectsDetection,
  notebookUrl,
  cover,
};

export default experiment;
