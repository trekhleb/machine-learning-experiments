// @flow
import React from 'react';
import { Router } from 'react-router-dom';
import { createHashHistory } from 'history';
import type { Location, BrowserHistory } from 'history';

import Layout from './components/shared/Layout';
import Routes from './Routes';
import { googleAnalyticsTrack } from './utils/analytics';

const history: BrowserHistory = createHashHistory();

history.listen((location: Location): void => {
  googleAnalyticsTrack(location);
});

function App() {
  return (
    <Router history={history}>
      <Layout>
        <Routes />
      </Layout>
    </Router>
  );
}

export default App;
