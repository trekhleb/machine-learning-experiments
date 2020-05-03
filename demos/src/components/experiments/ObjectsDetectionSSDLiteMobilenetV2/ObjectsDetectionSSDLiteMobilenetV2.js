// @flow
import React, {
  useState,
  useEffect,
  useRef,
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
import CanvasShapes from '../../shared/CanvasShapes';
import CameraStream from '../../shared/CameraStream';
import Snack from '../../shared/Snack';
import CocoClasses from './classes';
import type { Box as BoxType } from '../../shared/CanvasShapes';

import cover from '../../../images/objects_detection_ssdlite_mobilenet_v2.jpg';
import useGraphModel from '../../../hooks/useGraphModel';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.ObjectsDetectionSSDLiteMobilenetV2;
const experimentName = 'Objects Detection (MobileNetV2)';
const experimentDescription = 'Detecting objects in your camera stream (SSDLite, Mobilenet V2, COCO database, 90 object categories)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/objects_detection_ssdlite_mobilenet_v2/model.json`;
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

const ObjectsDetectionSSDLiteMobilenetV2 = (): Node => {
  const classes = useStyles();

  const cameraStreamWrapper = useRef(null);

  const { model, modelErrorMessage } = useGraphModel({
    modelPath,
    warmup: true,
  });
  const [width, setWidth] = useState(defaultCameraStreamSize);
  const [height, setHeight] = useState(defaultCameraStreamSize);
  const [boxes, setBoxes] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

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

    const numDetections = numDetectionsTensor.arraySync().pop();
    const detectionScores = scoresTensor.arraySync().pop();
    const detectionClasses = classesTensor.toInt().arraySync().pop();
    const detectionBoxes = boxesTensor.arraySync().pop();

    if (!detectionScores || !detectionClasses || !detectionBoxes) {
      setErrorMessage('Cannot extract detection info');
    }

    const indicesTensor = await tf.image.nonMaxSuppressionAsync(
      detectionBoxes, detectionScores, maxNumBoxes, iouThreshold, scoreThreshold,
    );

    const importantDetectionIndices = indicesTensor.dataSync();

    const detections: BoxType[] = [];

    for (let detectionIndex = 0; detectionIndex < numDetections; detectionIndex += 1) {
      if (importantDetectionIndices.includes(detectionIndex)) {
        const cocoClassId = detectionClasses[detectionIndex];
        const cocoClass = CocoClasses[detectionClasses[`${detectionIndex}`]];

        if (!cocoClass) {
          setErrorMessage(`Unknown COCO class ID: ${cocoClassId} ${detectionClasses.slice(0, 4).toString()}`);
        }

        const cocoClassName = cocoClass ? cocoClass.displayName : '';

        const score = Math.floor(100 * detectionScores[detectionIndex]);
        const detectionBox = detectionBoxes[detectionIndex];

        const x1 = Math.floor(inputImageHeight * detectionBox[1]);
        const y1 = Math.floor(inputImageWidth * detectionBox[0]);
        const x2 = Math.floor(inputImageHeight * detectionBox[3]);
        const y2 = Math.floor(inputImageWidth * detectionBox[2]);

        const detection: BoxType = {
          x: x1,
          y: y1,
          width: x2 - x1,
          height: y2 - y1,
          label: `${cocoClassName} ${score}%`,
        };

        detections.push(detection);
      }
    }

    setBoxes(detections);

    return true;
  };

  const onVideoFrame = async (video?: ?HTMLVideoElement) => {
    await executeModel(video);
  };

  useEffect(() => {
    if (cameraStreamWrapper.current && cameraStreamWrapper.current.offsetWidth) {
      const size = Math.min(maxCameraStreamSize, cameraStreamWrapper.current.offsetWidth);
      setWidth(size);
      setHeight(size);
    }
  }, []);

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

  const notificationElement = (
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
      <Snack severity="error" message={errorMessage || modelErrorMessage} />
    </Box>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: ObjectsDetectionSSDLiteMobilenetV2,
  notebookUrl,
  cover,
};

export default experiment;
