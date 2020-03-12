// @flow
import type { ExperimentsMap } from './types';
import DigitsRecognitionMLP from './DigitsRecognitionMLP/DigitsRecognitionMLP';
import DigitsRecognitionCNN from './DigitsRecognitionCNN/DigitsRecognitionCNN';
import ObjectsDetectionSSDLiteMobilenetV2 from './ObjectsDetectionSSDLiteMobilenetV2/ObjectsDetectionSSDLiteMobilenetV2';
import ImageClassificationMobilenetV2 from './ImageClassificationMobilenetV2/ImageClassificationMobilenetV2';
import RockPaperScissorsCNN from './RockPaperScissorsCNN/RockPaperScissorsCNN';
import RockPaperScissorsMobilenetV2 from './RockPaperScissorsMobilenetV2/RockPaperScissorsMobilenetV2';

const experiments: ExperimentsMap = {
  DigitsRecognitionMLP,
  DigitsRecognitionCNN,
  RockPaperScissorsCNN,
  RockPaperScissorsMobilenetV2,
  ObjectsDetectionSSDLiteMobilenetV2,
  ImageClassificationMobilenetV2,
};

export default experiments;
