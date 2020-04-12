// @flow
import { useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs-layers';

type UseLayersModelProps = {
  modelPath: string,
  warmup?: boolean,
};

type UseLayersModelOutput = {
  model: ?LayersModel,
  modelErrorMessage: ?string,
};

const useLayersModel = (props: UseLayersModelProps): UseLayersModelOutput => {
  const { modelPath, warmup = false } = props;

  const [model, setModel] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
  const [modelErrorMessage, setModelErrorMessage] = useState(null);

  const warmupModel = async () => {
    if (warmup && model && !modelIsWarm) {
      const inputShapeWithNulls = model.input.shape;
      const inputShape = inputShapeWithNulls.map((dimension) => {
        if (dimension === null || dimension === -1) {
          return 1;
        }
        return dimension;
      });

      const fakeInput = tf.zeros(inputShape, 'int32');
      model.predict(fakeInput);
    }
  };

  const warmupModelCallback = useCallback(warmupModel, [model, modelIsWarm]);

  // Effect for loading the model.
  useEffect(() => {
    tf.loadLayersModel(modelPath)
      .then((layersModel: LayersModel) => {
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

  let finalModel: ?LayersModel = model;
  if (warmup) {
    finalModel = modelIsWarm ? model : null;
  }

  return {
    model: finalModel,
    modelErrorMessage,
  };
};

export default useLayersModel;
