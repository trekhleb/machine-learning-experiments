import React from 'react';
import type { Node } from 'react';
import { withRouter } from 'react-router-dom';
import type { Match } from 'react-router-dom';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';

import type { Experiment as ExperimentType } from '../experiments/types';
import experiments from '../experiments';
import { EXPERIMENT_ID_PARAM } from '../../constants/routes';

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
        <Button size="small" variant="outlined" href={experiment.trainingURL}>
          See how this model was trained
        </Button>
      </Box>
      <Box mb={1}>
        <ExperimentElement />
      </Box>
    </>
  );
};

export default withRouter(Experiment);
