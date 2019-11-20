import React from 'react';
import {Switch, Route} from 'react-router-dom';

import DigitsRecognitionExperiment from '../experiments/DigitsRecognitionExperiment/DigitsRecognitionExperiment';
import {EXPERIMENT_DIGITS_RECOGNITION_ROUTE, HOME_ROUTE} from '../../constants/routes';
import ExperimentsList from './ExperimentsList';

const Routes = () => {
  return (
    <Switch>
      <Route path={HOME_ROUTE} exact>
        <ExperimentsList />
      </Route>
      <Route path={EXPERIMENT_DIGITS_RECOGNITION_ROUTE} exact>
        <DigitsRecognitionExperiment />
      </Route>
    </Switch>
  );
};

export default Routes;
