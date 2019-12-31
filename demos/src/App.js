import React from 'react';
import { Router } from 'react-router-dom';
import { createBrowserHistory } from 'history';
import type { Location, BrowserHistory } from 'history';

import Layout from './components/shared/Layout';
import Routes from './Routes';
import { googleAnalyticsTrack } from './utils/analytics';

const history: BrowserHistory = createBrowserHistory();

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
