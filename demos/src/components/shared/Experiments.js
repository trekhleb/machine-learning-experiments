// @flow
import React from 'react';
import type { Node } from 'react';
import { Helmet } from 'react-helmet';
import Grid from '@material-ui/core/Grid';

import ExperimentPreview from './ExperimentPreview';
import experiments from '../experiments';
import type { Experiment } from '../experiments/types';
import { WINDOW_TITLE } from '../../constants/copies';
import ErrorBoundary from './ErrorBoundary';

const Experiments = (): Node => {
  const experimentsPreviews: Node[] = [];

  Object.keys(experiments).forEach((experimentId: string) => {
    const experiment: Experiment = experiments[experimentId];

    experimentsPreviews.push(
      <Grid item key={experimentId} xs={12} sm={6} lg={3}>
        <ExperimentPreview experiment={experiment} />
      </Grid>,
    );
  });

  return (
    <ErrorBoundary>
      <Helmet>
        <title>{WINDOW_TITLE}</title>
      </Helmet>
      <Box>
        Blah blah blah
      </Box>
      <Grid container spacing={3}>
        {experimentsPreviews}
      </Grid>
    </ErrorBoundary>
  );
};

export default Experiments;
