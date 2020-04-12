// @flow
import { useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { GraphModel } from '@tensorflow/tfjs-converter';

type UseGraphModelProps = {
  modelPath: string,
  warmup?: boolean,
};

type UseGraphModelOutput = {
  model: ?GraphModel,
  modelErrorMessage: ?string,
};

const useGraphModel = (props: UseGraphModelProps): UseGraphModelOutput => {
  const { modelPath, warmup = false } = props;

  const [model, setModel] = useState(null);
  const [modelIsWarm, setModelIsWarm] = useState(null);
  const [modelErrorMessage, setModelErrorMessage] = useState(null);

  const warmupModel = async () => {
    if (warmup && model && !modelIsWarm) {
      const inputShapeWithNulls = model.inputs[0].shape;
      const inputShape = inputShapeWithNulls.map((dimension) => {
        if (dimension === null || dimension === -1) {
          return 1;
        }
        return dimension;
      });

      const fakeInput = tf.zeros(inputShape, 'int32');
      await model.executeAsync(fakeInput);
    }
  };

  const warmupModelCallback = useCallback(warmupModel, [model, modelIsWarm]);

  // Effect for loading the model.
  useEffect(() => {
    tf.loadGraphModel(modelPath)
      .then((graphModel: GraphModel) => {
        setModel(graphModel);
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

  let finalModel: ?GraphModel = model;
  if (warmup) {
    finalModel = modelIsWarm ? model : null;
  }

  return {
    model: finalModel,
    modelErrorMessage,
  };
};

export default useGraphModel;
