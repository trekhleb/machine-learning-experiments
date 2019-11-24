import React from 'react';
import Typography from '@material-ui/core/Typography';

import cover from './cover.png';

const DigitsRecognition = () => {
  return (
    <Typography variant="h5">
      Digits Recognition
    </Typography>
  );
};

const experiment = {
  name: 'Digits Recognition',
  description: 'Hand-written digits recognition',
  component: DigitsRecognition,
  cover,
};

export default experiment;
