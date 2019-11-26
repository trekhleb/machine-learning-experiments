import React from 'react';
import { withRouter } from 'react-router-dom';
import type { Match } from 'react-router-dom';
import Box from '@material-ui/core/Box';
import Typography from '@material-ui/core/Typography';

import type { Experiment as ExperimentType } from '../experiments/types';
import experiments from '../experiments';
import { EXPERIMENT_ID_PARAM } from '../../constants/routes';

type ExperimentProps = {
  match: Match,
};

const Experiment = (props: ExperimentProps) => {
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
      <Typography variant="h5">
        {experiment.name}
      </Typography>
      <Typography variant="body1" component="p">
        {experiment.description}
      </Typography>
      <Box marginTop={3}>
        <ExperimentElement />
      </Box>
    </>
  );
};

export default withRouter(Experiment);
