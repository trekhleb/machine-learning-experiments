import React from 'react';
import type { Node } from 'react';
import { withRouter } from 'react-router-dom';
import type { Match } from 'react-router-dom';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';

import type { Experiment as ExperimentType } from '../experiments/types';
import experiments from '../experiments';
import { EXPERIMENT_ID_PARAM } from '../../constants/routes';
import { MACHINE_LEARNING_EXPERIMENTS_GITHUB_URL } from '../../constants/links';
import Badge, { badgeType } from './Badge';

type ExperimentProps = {
  match: Match,
};

const Experiment = (props: ExperimentProps): Node => {
  const { match } = props;

  const experimentId: ?string = match.params[EXPERIMENT_ID_PARAM];

  const unknownExperiment = <div>Unknown experiment</div>;

  if (!experimentId) {
    return unknownExperiment;
  }

  if (!Object.prototype.hasOwnProperty.call(experiments, experimentId)) {
    return unknownExperiment;
  }

  const experiment: ExperimentType = experiments[experimentId];
  const ExperimentElement = experiment.component;

  const githubLink = (
    <Grid item>
      <Badge
        url={MACHINE_LEARNING_EXPERIMENTS_GITHUB_URL}
        type={badgeType.github}
      />
    </Grid>
  );

  const colabLink = experiment.colabURL ? (
    <Grid item>
      <Badge
        url={experiment.colabURL}
        type={badgeType.colab}
      />
    </Grid>
  ) : null;

  const jupyterLink = experiment.jupyterURL ? (
    <Grid item>
      <Badge
        url={experiment.jupyterURL}
        type={badgeType.jupyter}
      />
    </Grid>
  ) : null;

  return (
    <>
      <Box mb={1}>
        <Typography variant="h5">
          {experiment.name}
        </Typography>
      </Box>
      <Box mb={1}>
        <Typography variant="body1" component="p">
          {experiment.description}
        </Typography>
      </Box>
      <Box mb={3}>
        <Grid container spacing={1} alignItems="center" justify="flex-start">
          {colabLink}
          {jupyterLink}
          {githubLink}
        </Grid>
      </Box>
      <Box mb={3}>
        <ExperimentElement />
      </Box>
    </>
  );
};

export default withRouter(Experiment);
