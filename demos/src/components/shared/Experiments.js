import React from 'react';
import Grid from '@material-ui/core/Grid';

import ExperimentPreview from './ExperimentPreview';
import experiments from '../experiments';

const Experiments = () => {
  const experimentsPreviews = [];

  for (let experimentId in experiments) {
    const experiment = experiments[experimentId];

    experimentsPreviews.push(
      <Grid item key={experimentId} xs={12} sm={6} lg={3}>
        <ExperimentPreview
          id={experimentId}
          name={experiment.name}
          cover={experiment.cover}
          description={experiment.description}
        />
      </Grid>
    );
  }

  return (
    <Grid container spacing={3}>
      {experimentsPreviews}
    </Grid>
  );
};

export default Experiments;
