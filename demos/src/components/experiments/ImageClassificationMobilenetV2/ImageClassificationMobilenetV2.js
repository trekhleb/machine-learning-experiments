import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import type { Node } from 'react';
import Box from '@material-ui/core/Box';
import { makeStyles } from '@material-ui/core/styles';

import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';
import cover from './cover.png';
import Snack from '../../shared/Snack';
import ImageInput from '../../shared/ImageInput';

const experimentSlug = 'ImageClassificationMobilenetV2';
const experimentName = 'Image Classification (MobileNetV2)';
const experimentDescription = 'Generate classification tags for the images (Mobilenet V2, ImageNet database, 1000 object categories)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/image_classification_mobilenet_v2/model.json`;

const useStyles = makeStyles(() => ({
  paper: {
    overflow: 'hidden',
  },
}));

const ImageClassificationMobilenetV2 = (): Node => {
  const classes = useStyles();

  const [model, setModel] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Load the model.
  useEffect(() => {
    if (model) {
      return;
    }

    tf.loadLayersModel(modelPath)
      .then((layersModel) => {
        setModel(layersModel);
      })
      .catch((e) => {
        setErrorMessage('Model cannot be loaded');
      });
  }, [model, setErrorMessage, setModel]);

  return (
    <Box>
      <ImageInput />
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
