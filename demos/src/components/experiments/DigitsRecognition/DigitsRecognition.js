import React from 'react';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';

import Canvas from '../../shared/Canvas';
import type { Experiment } from '../types';
import cover from './cover.png';

const experimentSlug = 'DigitsRecognition';
const experimentName = 'Digits Recognition';
const experimentDescription = 'Hand-written digits recognition';

const DigitsRecognition = (): Node => (
  <Paper>
    <Canvas />
  </Paper>
);

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: DigitsRecognition,
  cover,
};

export default experiment;
