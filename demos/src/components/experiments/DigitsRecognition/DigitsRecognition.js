import React from 'react';
import type { Node } from 'react';

import type { Experiment } from '../types';
import cover from './cover.png';

const experimentSlug = 'DigitsRecognition';
const experimentName = 'Digits Recognition';
const experimentDescription = 'Hand-written digits recognition';

const DigitsRecognition = (): Node => (
  <>
      Canvas goes here
  </>
);

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: DigitsRecognition,
  cover,
};

export default experiment;
