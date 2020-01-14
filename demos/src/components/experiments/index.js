import type { ExperimentsMap } from './types';
import DigitsRecognitionMLP from './DigitsRecognitionMLP/DigitsRecognitionMLP';
import DigitsRecognitionCNN from './DigitsRecognitionCNN/DigitsRecognitionCNN';
import ObjectsDetection from './ObjectsDetection/ObjectsDetection';

const experiments: ExperimentsMap = {
  DigitsRecognitionMLP,
  DigitsRecognitionCNN,
  ObjectsDetection,
};

export default experiments;
