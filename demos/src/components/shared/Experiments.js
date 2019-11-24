import React from 'react';
import Grid from '@material-ui/core/Grid';

import ExperimentPreview from './ExperimentPreview';
import experiments from '../experiments';

const Experiments = () => {
  const experimentsPreviews = [];

  for (let experimentId in experiments) {
    const experiment = experiments[experimentId];

    experimentsPreviews.push(
      <ExperimentPreview
        id={experimentId}
        key={experimentId}
        name={experiment.name}
        cover={experiment.cover}
        description={experiment.description}
      />
    );
  }

  return <Grid>{experimentsPreviews}</Grid>;
};

export default Experiments;
