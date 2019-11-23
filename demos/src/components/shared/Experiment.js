import React from 'react';
import {withRouter} from 'react-router-dom';

import experiments from '../experiments';
import {EXPERIMENT_ID_PARAM} from '../../constants/routes';

const Experiment = (props) => {
  const {match} = props;

  const experimentId = match.params[EXPERIMENT_ID_PARAM];

  if (!Object.prototype.hasOwnProperty.call(experiments, experimentId)) {
    return <div>Unknown experiment</div>;
  }

  const ExperimentElement = experiments[experimentId].component;

  return <ExperimentElement />;
};

export default withRouter(Experiment);
