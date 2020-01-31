import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
} from 'react';
import * as tf from '@tensorflow/tfjs';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';
import { makeStyles } from '@material-ui/core/styles';
import LinearProgress from '@material-ui/core/LinearProgress';
import PhoneIphoneIcon from '@material-ui/icons/PhoneIphone';
import Paper from '@material-ui/core/Paper';

import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';
import cover from './cover.png';
import CanvasShapes from '../../shared/CanvasShapes';
import CameraStream from '../../shared/CameraStream';
import Snack from '../../shared/Snack';
import type { Box as BoxType } from '../../shared/CanvasShapes';

const experimentSlug = 'ImageClassificationMobilenetV2';
const experimentName = 'Image Classification (MobileNetV2)';
const experimentDescription = 'Generate classification tags for the images (Mobilenet V2, ImageNet database, 1000 object categories)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/image_classification_mobilenet_v2/model.json`;
const defaultCameraStreamSize = 300;
const maxCameraStreamSize = 400;

const maxNumBoxes = 20;
const iouThreshold = 0.5;
const scoreThreshold = 0.2;

const useStyles = makeStyles(() => ({
  paper: {
    overflow: 'hidden',
  },
}));

const ImageClassificationMobilenetV2 = (): Node => {
  const classes = useStyles();

  const cameraStreamWrapper = useRef(null);

  const [width, setWidth] = useState(defaultCameraStreamSize);
  const [height, setHeight] = useState(defaultCameraStreamSize);
  const [model, setModel] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
  const [boxes, setBoxes] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const warmupModel = async () => {
    if (model && !modelIsWarm) {
      const result = await model.executeAsync(tf.zeros([1, 300, 300, 3]));
      await Promise.all(result.map((tensor) => tensor.data()));
      result.map((tensor) => tensor.dispose());
    }
  };

  const executeModel = async (video?: ?HTMLVideoElement) => {
    if (!model || !video) {
      return false;
    }

    const tensor = tf.browser.fromPixels(video);
    const inputImageWidth = tensor.shape[0];
    const inputImageHeight = tensor.shape[1];

    let result;
    try {
      result = await model.executeAsync(tensor.expandDims(0));
    } catch (e) {
      let message = 'Video cannot be started';
      if (e && e.message) {
        message += `: ${e.message}`;
      }
      setErrorMessage(message);
    }

    if (!result) {
      return false;
    }

    const scoresTensor = result[0];
    const classesTensor = result[1];
    const numDetectionsTensor = result[2];
    const boxesTensor = result[3];

    if (!scoresTensor || !classesTensor || !numDetectionsTensor || !boxesTensor) {
      setErrorMessage('Detection tensors not found');
      return false;
    }

    return true;
  };

  const onVideoFrame = async (video?: ?HTMLVideoElement) => {
    await executeModel(video);
  };

  const warmupModelCallback = useCallback(warmupModel, [model, modelIsWarm]);

  // Load the model.
  useEffect(() => {
    if (cameraStreamWrapper.current && cameraStreamWrapper.current.offsetWidth) {
      const size = Math.min(maxCameraStreamSize, cameraStreamWrapper.current.offsetWidth);
      setWidth(size);
      setHeight(size);
    }

    if (model) {
      return;
    }

    tf.loadGraphModel(modelPath)
      .then((graphModel) => {
        setModel(graphModel);
      })
      .catch((e) => {
        setErrorMessage('Model cannot be loaded');
      });
  }, [model, setErrorMessage, setModel]);

  // Warmup the model.
  useEffect(() => {
    if (model && !modelIsWarm) {
      warmupModelCallback().then(() => setModelIsWarm(true));
    }
  }, [model, modelIsWarm, setModelIsWarm, warmupModelCallback]);

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

  const notificationElement = modelIsWarm === null ? (
    <>
      Preparing the model...
    </>
  ) : (
    <>
      <PhoneIphoneIcon />
      {' '}
      Your camera stream appears here
    </>
  );

  return (
    <Box ref={cameraStreamWrapper}>
      <Box mb={1} display="flex" alignItems="center">
        {notificationElement}
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
  component: ImageClassificationMobilenetV2,
  notebookUrl,
  cover,
};

export default experiment;
