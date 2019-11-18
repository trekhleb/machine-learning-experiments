import React from 'react';
import {BrowserRouter as Router, Switch, Route, Link} from 'react-router-dom';

import DigitsRecognitionExperiment from './experiments/digitsRecognition/DigitsRecognitionExperiment';
import FaceRecognitionExperiment from './experiments/faceRecognition/FaceRecognitionExperiment';
import './App.css';

function App() {
  return (
    <Router>
      <div>
        <h1>Hello</h1>

        <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/experiment/digits_recognition">Digits recognition experiment</Link>
          </li>
          <li>
            <Link to="/experiment/face_recognition">Face recognition experiment</Link>
          </li>
        </ul>

        <Switch>
          <Route path="/experiment/digits_recognition" exact>
            <DigitsRecognitionExperiment />
          </Route>
          <Route path="/experiment/face_recognition" exact>
            <FaceRecognitionExperiment />
          </Route>
        </Switch>
      </div>
    </Router>
  );
}

export default App;
