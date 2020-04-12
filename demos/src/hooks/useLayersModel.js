// @flow
import { useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';

type UseLayersModelProps = {
  modelPath: string,
  warmup?: boolean,
};

type UseLayersModelOutput = {
  model: ?any,
  modelErrorMessage: ?string,
};

const useLayersModel = (props: UseLayersModelProps): UseLayersModelOutput => {
  const { modelPath, warmup = false } = props;

  const [model, setModel] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
  const [modelErrorMessage, setModelErrorMessage] = useState(null);

  const warmupModel = async () => {
    if (warmup && model && !modelIsWarm) {
      // const result = await model.executeAsync(tf.zeros([1, 300, 300, 3]));
      // await Promise.all(result.map((tensor) => tensor.data()));
      // result.map((tensor) => tensor.dispose());

      // const modelInputWidth = model.layers[0].input.shape[1];
      // const modelInputHeight = model.layers[0].input.shape[2];
      // model.predict(
      //   tf.zeros([1, modelInputWidth, modelInputHeight, 3]),
      // );

      // const modelInputWidth = model.input.shape[1];
      // const modelInputHeight = model.input.shape[2];
      // const fakeInput = tf.zeros([1, modelInputWidth, modelInputHeight, 3]);

      const inputShapeWithNulls = model.input.shape;
      const inputShape = inputShapeWithNulls.map((dimension) => {
        if (dimension === null) {
          return 1;
        }
        return dimension;
      });

      const fakeInput = tf.zeros(inputShape);
      model.predict(fakeInput);
    }
  };

  const warmupModelCallback = useCallback(warmupModel, [model, modelIsWarm]);

  // Effect for loading the model.
  useEffect(() => {
    tf.loadLayersModel(modelPath)
      .then((layersModel) => {
        setModel(layersModel);
      })
      .catch((e) => {
        setModelErrorMessage(e.message);
      });
  }, [modelPath, setModelErrorMessage, setModel]);

  // Effect for warming up a model.
  useEffect(() => {
    if (warmup && model && !modelIsWarm) {
      warmupModelCallback().then(() => {
        setModelIsWarm(true);
      });
    }
  }, [
    model,
    warmup,
    modelIsWarm,
    setModelIsWarm,
    warmupModelCallback,
  ]);

  let finalModel = model;
  if (warmup) {
    finalModel = modelIsWarm ? model : null;
  }

  return {
    model: finalModel,
    modelErrorMessage,
  };
};

export default useLayersModel;
