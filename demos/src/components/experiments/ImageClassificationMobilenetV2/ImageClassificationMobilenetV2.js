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
import cover from './cover.png';
import Snack from '../../shared/Snack';
import ImageInput from '../../shared/ImageInput';
import imageNetLabels from './imageNetLabels.json';

const experimentSlug = 'ImageClassificationMobilenetV2';
const experimentName = 'Image Classification (MobileNetV2)';
const experimentDescription = 'Generate classification tags for the images (Mobilenet V2, ImageNet database, 1000 object categories)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/image_classification_mobilenet_v2/image_classification_mobilenet_v2.ipynb`;

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

  const experimentWrapper = useRef(null);
  const imagePreviewRef = useRef(null);
  const [model, setModel] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
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
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight]);

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

  const warmupModel = async () => {
    if (model && !modelIsWarm) {
      const modelInputWidth = model.layers[0].input.shape[1];
      const modelInputHeight = model.layers[0].input.shape[2];
      model.predict(
        tf.zeros([1, modelInputWidth, modelInputHeight, 3]),
      );
    }
  };

  const warmupModelCallback = useCallback(warmupModel, [model, modelIsWarm]);

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

  // Warmup the model.
  useEffect(() => {
    if (model && !modelIsWarm) {
      warmupModelCallback();
      setModelIsWarm(true);
    }
  }, [model, modelIsWarm, setModelIsWarm, warmupModelCallback]);

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

  const notificationElement = modelIsWarm === null ? (
    <>Preparing the model...</>
  ) : (
    <>Select an image or take a photo that you want to be tagged (classified).</>
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
      <ImageInput onSelect={setImages} disabled={!modelIsWarm} />
      {tagsContainer}
      {tagsError}
      <Box mt={2}>
        {imagesPreview}
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
