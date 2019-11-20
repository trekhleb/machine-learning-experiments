import React from 'react';
import Grid from '@material-ui/core/Grid';

import ExperimentPreview from './ExperimentPreview';

const ExperimentsList = () => {
  return (
    <Grid>
      <ExperimentPreview name="Digits recognition" id="digits_recognition" />
    </Grid>
  );
};

export default ExperimentsList;
