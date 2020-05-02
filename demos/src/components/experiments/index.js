// @flow
import type { ExperimentsMap } from './types';
import DigitsRecognitionMLP from './DigitsRecognitionMLP/DigitsRecognitionMLP';
import SketchRecognitionMLP from './SketchRecognitionMLP/SketchRecognitionMLP';
import DigitsRecognitionCNN from './DigitsRecognitionCNN/DigitsRecognitionCNN';
import SketchRecognitionCNN from './SketchRecognitionCNN/SketchRecognitionCNN';
import ObjectsDetectionSSDLiteMobilenetV2 from './ObjectsDetectionSSDLiteMobilenetV2/ObjectsDetectionSSDLiteMobilenetV2';
import ImageClassificationMobilenetV2 from './ImageClassificationMobilenetV2/ImageClassificationMobilenetV2';
import RockPaperScissorsCNN from './RockPaperScissorsCNN/RockPaperScissorsCNN';
import RockPaperScissorsMobilenetV2 from './RockPaperScissorsMobilenetV2/RockPaperScissorsMobilenetV2';
import TextGenerationShakespeareRNN from './TextGenerationShakespeareRNN/TextGenerationShakespeareRNN';
import TextGenerationWikipediaRNN from './TextGenerationWikipediaRNN/TextGenerationWikipediaRNN';
import NumbersSummationRNN from './NumbersSummationRNN/NumbersSummationRNN';

const experiments: ExperimentsMap = {
  DigitsRecognitionMLP,
  DigitsRecognitionCNN,
  SketchRecognitionMLP,
  SketchRecognitionCNN,
  RockPaperScissorsCNN,
  RockPaperScissorsMobilenetV2,
  ObjectsDetectionSSDLiteMobilenetV2,
  ImageClassificationMobilenetV2,
  NumbersSummationRNN,
  TextGenerationShakespeareRNN,
  TextGenerationWikipediaRNN,
};

export default experiments;
