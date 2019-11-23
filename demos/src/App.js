import React from 'react';
import {BrowserRouter} from 'react-router-dom';

import Layout from './components/shared/Layout';
import Routes from './Routes';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes />
      </Layout>
    </BrowserRouter>
  );
}

export default App;
