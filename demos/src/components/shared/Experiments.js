// @flow
import React from 'react';
import type { Node } from 'react';
import Typography from '@material-ui/core/Typography';
import { Helmet } from 'react-helmet';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';

import ExperimentPreview from './ExperimentPreview';
import experiments from '../experiments';
import type { Experiment } from '../experiments/types';
import { WINDOW_TITLE } from '../../constants/copies';
import ErrorBoundary from './ErrorBoundary';
import { ML_EXPERIMENTS_GITHUB_URL } from '../../constants/links';

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
      <Box mb={4}>
        <Typography variant="body1" gutterBottom>
          This is a demo app for
          {' '}
          <Link href={ML_EXPERIMENTS_GITHUB_URL} color="secondary">
            Machine Learning Experiments
          </Link>
          {' '}
          GitHub repository. Go there if you want to see how all these models were trained.
        </Typography>
      </Box>
      <Grid container spacing={3}>
        {experimentsPreviews}
      </Grid>
    </ErrorBoundary>
  );
};

export default Experiments;
