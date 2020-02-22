// @flow
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

const rootElement: ?Element = document.getElementById('root');
if (rootElement != null) {
  ReactDOM.render(<App />, rootElement);
}
