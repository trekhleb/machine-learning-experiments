import type { ExperimentsMap } from './types';
import DigitsRecognitionMLP from './DigitsRecognitionMLP/DigitsRecognitionMLP';
import DigitsRecognitionCNN from './DigitsRecognitionCNN/DigitsRecognitionCNN';

const experiments: ExperimentsMap = {
  DigitsRecognitionMLP,
  DigitsRecognitionCNN,
};

export default experiments;
