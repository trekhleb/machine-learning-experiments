// @flow
import React, {
  useState, useEffect, useRef, useCallback,
} from 'react';
import * as tf from '@tensorflow/tfjs';
import type { Node } from 'react';
import Chip from '@material-ui/core/Chip';
import HelpIcon from '@material-ui/icons/Help';
import LocalOfferIcon from '@material-ui/icons/LocalOffer';
import Box from '@material-ui/core/Box';
import { makeStyles } from '@material-ui/styles';

import {
  ML_EXPERIMENTS_DEMO_MODELS_PATH,
  ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL,
} from '../../../constants/links';
import type { Experiment } from '../types';
import Snack from '../../shared/Snack';
import ImageInput from '../../shared/ImageInput';
import imageNetLabels from './imageNetLabels.json';

import cover from '../../../images/image_classification_mobilenet_v2.jpg';
import inputImageExample0 from './input-examples/0.png';
import inputImageExample1 from './input-examples/1.png';
import inputImageExample2 from './input-examples/2.png';
import inputImageExample3 from './input-examples/3.png';
import inputImageExample4 from './input-examples/4.png';
import inputImageExample5 from './input-examples/5.png';
import useLayersModel from '../../../hooks/useLayersModel';
import { experimentsSlugs } from '../types';

const experimentSlug = experimentsSlugs.ImageClassificationMobilenetV2;
const experimentName = 'Image Classification (MobileNetV2)';
const experimentDescription = 'Generate classification tags for the images (Mobilenet V2, ImageNet database, 1000 object categories)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb`;
const inputImagesExamples = [
  inputImageExample0,
  inputImageExample1,
  inputImageExample2,
  inputImageExample3,
  inputImageExample4,
  inputImageExample5,
];

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/image_classification_mobilenet_v2/model.json`;

const maxPreviewWidth = 400;

type Prediction = {
  label: string,
  probability: number,
};

const predictionThreshold = 0.2;
const tagsLimit = 5;

const useStyles = makeStyles((theme) => ({
  tagsContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    '& > *': {
      marginRight: theme.spacing(1),
      marginBottom: theme.spacing(1),
    },
  },
}));

const ImageClassificationMobilenetV2 = (): Node => {
  const classes = useStyles();

  const { model, modelErrorMessage } = useLayersModel({
    modelPath,
    warmup: true,
  });
  const experimentWrapper = useRef(null);
  const imagePreviewRef = useRef(null);
  const [images, setImages] = useState(null);
  const [previewWidth, setPreviewWidth] = useState(maxPreviewWidth);
  const [predictions, setPredictions] = useState([]);

  const classifyImage = () => {
    if (!model) {
      return;
    }

    const image = imagePreviewRef.current;
    if (!image) {
      return;
    }

    const modelInputWidth = model.layers[0].input.shape[1];
    const modelInputHeight = model.layers[0].input.shape[2];

    const tensor: tf.Tensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight])
      // Imitating tf.keras.applications.mobilenet_v2.preprocess_input() function.
      // Trying to convert [0, 255] range to [-1, 1] range.
      // @see: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L18
      .div(255 / 2)
      .add(-1);

    const batchAxis = 0;
    const currentPredictions = model.predict(tensor.expandDims(batchAxis));
    const predictionsArray = currentPredictions.arraySync()[0];

    const predictionsSet: Prediction[] = imageNetLabels.map((label: string, index: number) => {
      const prediction: Prediction = {
        label,
        probability: predictionsArray[index],
      };
      return prediction;
    });

    setPredictions(predictionsSet);
  };

  const classifyImageCallback = useCallback(
    classifyImage,
    [model],
  );

  // Setup preview width.
  useEffect(() => {
    if (!experimentWrapper.current) {
      return;
    }
    const width = Math.min(
      maxPreviewWidth,
      experimentWrapper.current.offsetWidth,
    );
    setPreviewWidth(width);
  }, []);

  // Classify an image if new image has been selected.
  useEffect(() => {
    if (!images || !images.length || !imagePreviewRef.current) {
      return () => {};
    }
    const imagePreview = imagePreviewRef.current;
    imagePreview.addEventListener('load', classifyImageCallback, { once: true });
    return () => {
      imagePreview.removeEventListener('load', classifyImageCallback, { once: true });
    };
  }, [images, classifyImageCallback]);

  const notificationElement = (
    <Box display="flex">
      Select an image or take a photo that you want to be tagged (classified).
      Try to have one distinct object on the photo.
    </Box>
  );

  const imagesPreview = images ? (
    images.map((image: File) => (
      <Box
        key={image.name}
        boxShadow={2}
        width={previewWidth}
        fontSize={0}
      >
        <img
          ref={imagePreviewRef}
          src={URL.createObjectURL(image)}
          alt={image.name}
          width={previewWidth}
        />
      </Box>
    ))
  ) : null;

  const tags = predictions && predictions.length
    ? predictions
      .filter((prediction: Prediction) => prediction.probability > predictionThreshold)
      .sort((a: Prediction, b: Prediction) => {
        if (a.probability === b.probability) { return 0; }
        if (a.probability < b.probability) { return 1; }
        return -1;
      })
      .filter((prediction: Prediction, index) => index < tagsLimit)
      .map((prediction: Prediction) => {
        const label = `${prediction.label} | ${Math.floor(100 * prediction.probability)}%`;
        return (
          <Chip
            key={prediction.label}
            label={label}
            color="secondary"
            icon={<LocalOfferIcon />}
          />
        );
      })
    : null;

  const tagsContainer = tags && tags.length ? (
    <Box className={classes.tagsContainer} mt={2} mb={2}>
      {tags}
    </Box>
  ) : null;

  const tagsError = predictions && predictions.length && (!tags || !tags.length) ? (
    <Box className={classes.tagsContainer} mt={2} mb={2}>
      <Chip
        label="Cannot classify"
        icon={<HelpIcon />}
      />
    </Box>
  ) : null;

  return (
    <Box ref={experimentWrapper}>
      <Box mb={2}>
        {notificationElement}
      </Box>
      <ImageInput onSelect={setImages} disabled={!model} />
      {tagsContainer}
      {tagsError}
      <Box mt={2}>
        {imagesPreview}
      </Box>
      <Snack severity="error" message={modelErrorMessage} />
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
  inputImageExamples: {
    imageWidth: 87,
    images: inputImagesExamples,
  },
};

export default experiment;
