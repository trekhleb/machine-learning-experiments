// @flow
import React from 'react';
import { Switch, Route } from 'react-router-dom';

import { HOME_ROUTE, EXPERIMENT_ROUTE } from './constants/routes';
import Experiments from './components/shared/Experiments';
import Experiment from './components/shared/Experiment';

const Routes = () => (
  <Switch>
    <Route path={HOME_ROUTE} exact>
      <Experiments />
    </Route>
    <Route path={EXPERIMENT_ROUTE} exact>
      <Experiment />
    </Route>
  </Switch>
);

export default Routes;
