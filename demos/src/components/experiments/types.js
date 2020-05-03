// @flow
import type { Node } from 'react';

export const experimentsSlugs = {
  DigitsRecognitionMLP: 'DigitsRecognitionMLP',
  DigitsRecognitionCNN: 'DigitsRecognitionCNN',
  SketchRecognitionMLP: 'SketchRecognitionMLP',
  SketchRecognitionCNN: 'SketchRecognitionCNN',
  RockPaperScissorsCNN: 'RockPaperScissorsCNN',
  RockPaperScissorsMobilenetV2: 'RockPaperScissorsMobilenetV2',
  ObjectsDetectionSSDLiteMobilenetV2: 'ObjectsDetectionSSDLiteMobilenetV2',
  ImageClassificationMobilenetV2: 'ImageClassificationMobilenetV2',
  NumbersSummationRNN: 'NumbersSummationRNN',
  TextGenerationShakespeareRNN: 'TextGenerationShakespeareRNN',
  TextGenerationWikipediaRNN: 'TextGenerationWikipediaRNN',
};

export type Experiment = {|
  slug: string,
  name: string,
  description: string,
  component: () => Node,
  cover: string,
  notebookUrl?: ?string,
  inputImageExamples?: {
    images: string[],
    imageWidth?: string | number,
  },
  inputTextExamples?: ?string[],
|};

export type ExperimentsMap = {
  [string]: Experiment,
};
