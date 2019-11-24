import React from 'react';
import {withRouter} from 'react-router-dom';
import Typography from '@material-ui/core/Typography';

import experiments from '../experiments';
import {EXPERIMENT_ID_PARAM} from '../../constants/routes';

const Experiment = (props) => {
  const {match} = props;

  const experimentId = match.params[EXPERIMENT_ID_PARAM];

  if (!Object.prototype.hasOwnProperty.call(experiments, experimentId)) {
    return <div>Unknown experiment</div>;
  }

  const experiment = experiments[experimentId];
  const ExperimentElement = experiment.component;

  return (
    <>
      <Typography variant="h5">
        {experiment.name}
      </Typography>
      <Typography variant="body1" component="p">
        {experiment.description}
      </Typography>
      <ExperimentElement />
    </>
  );
};

export default withRouter(Experiment);
