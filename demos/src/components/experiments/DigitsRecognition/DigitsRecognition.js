import React from 'react';
import type {Node} from 'react';
import Typography from '@material-ui/core/Typography';

import type {ExperimentProps, Experiment} from '../types'; 
import cover from './cover.png';

const experimentName = 'Digits Recognition';
const experimentDescription = 'Hand-written digits recognition';

const DigitsRecognition = (props: ExperimentProps): Node => {
  return (
    <>
      Canvas goes here
    </>
  );
};

const experiment: Experiment = {
  name: experimentName,
  description: experimentDescription,
  component: DigitsRecognition,
  cover,
};

export default experiment;
