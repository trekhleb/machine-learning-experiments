// @flow
import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
} from 'react';
import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs-layers';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import LinearProgress from '@material-ui/core/LinearProgress';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import Grid from '@material-ui/core/Grid';
import Fab from '@material-ui/core/Fab';
import Zoom from '@material-ui/core/Zoom';

import CameraStream from './CameraStream';
import Snack from './Snack';
import OneHotBars from './OneHotBars';
import type { DataRecord } from './OneHotBars';

const flipVideoHorizontally = true;

const countDownStart = 3;
const countDownTimeout = 700;
const countDownHeight = 100;

const scoreDownWidth = 100;

const canvasWidth = 300;
const canvasHeight = 300;

const useStyles = makeStyles(() => ({
  description: {
    marginBottom: '20px',
  },
  paper: {
    width: canvasWidth,
    height: canvasHeight,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  paperSnapshot: {
    width: canvasWidth,
    height: canvasHeight,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: 'none',
    marginTop: `-${canvasHeight}px`,
    position: 'absolute',
    background: 'none',
    zIndex: 10,
  },
  paperHumanChoiceBackground: {
    width: canvasWidth,
    height: canvasHeight,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: 'none',
    marginTop: `-${canvasHeight}px`,
    position: 'absolute',
    background: 'rgba(255, 255, 255, .4)',
    zIndex: 20,
  },
  paperHumanChoice: {
    width: canvasWidth,
    height: canvasHeight,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: 'none',
    marginTop: `-${canvasHeight}px`,
    position: 'absolute',
    background: 'none',
    zIndex: 30,
  },
  choice: {
    width: canvasWidth,
    height: canvasHeight,
    fontSize: '150px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttons: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'stretch',
    width: `${scoreDownWidth}px`,
    height: `${countDownHeight}px`,
  },
  playerName: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    fontWeight: 450,
    marginBottom: '10px',
  },
  score: {
    fontSize: '35px',
    fontWeight: 400,
    width: `${scoreDownWidth}px`,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  countDown: {
    fontSize: '80px',
    fontWeight: 500,
    width: `${scoreDownWidth}px`,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: `${countDownHeight}px`,
  },
}));

const choiceIDs = {
  rock: 'rock',
  paper: 'paper',
  scissors: 'scissors',
};

type Choice = {
  id: $Values<typeof choiceIDs>,
  icon: string,
  beats: Array<$Values<typeof choiceIDs>>,
};

type Choices = {
  [choiceKey: $Values<typeof choiceIDs>]: Choice
};

const choices: Choices = {
  [choiceIDs.rock]: {
    id: choiceIDs.rock,
    icon: '✊',
    beats: [choiceIDs.scissors],
  },
  [choiceIDs.paper]: {
    id: choiceIDs.paper,
    icon: '✋',
    beats: [choiceIDs.rock],
  },
  [choiceIDs.scissors]: {
    id: choiceIDs.scissors,
    icon: '✌️',
    beats: [choiceIDs.paper],
  },
};

export type RockPaperScissorsProps = {
  model: ?LayersModel,
};

/* eslint-disable react/jsx-one-expression-per-line */
const RockPaperScissors = (props: RockPaperScissorsProps): Node => {
  const { model } = props;

  const [errorMessage, setErrorMessage] = useState(null);
  const [rawPredictions, setRawPredictions] = useState(null);
  const [computerChoice, setComputerChoice] = useState(null);
  const [humanChoice, setHumanChoice] = useState(null);
  const [humanScore, setHumanScore] = useState(0);
  const [computerScore, setComputerScore] = useState(0);
  const [counter, setCounter] = useState(null);
  const [videoFrame, setVideoFrame] = useState(null);
  const [canvasFlipped, setCanvasFlipped] = useState(false);

  const snapshotCanvasRef = useRef(null);

  const classes = useStyles();

  const predictHumanChoice = (currentCanvas: ?HTMLCanvasElement): ?Choice => {
    if (!model) {
      setErrorMessage('Model is not loaded');
      return null;
    }

    if (!currentCanvas) {
      setErrorMessage('Canvas is not rendered');
      return null;
    }

    const modelInputWidth = model.input.shape[1];
    const modelInputHeight = model.input.shape[2];

    const inputTensor = tf.browser
      .fromPixels(currentCanvas)
      // Resize image to fit neural network input.
      .resizeNearestNeighbor([modelInputWidth, modelInputHeight])
      // Normalize.
      .div(255);

    const prediction = model.predict(
      // Reshape and add one dimension for the pixel color to match CNN input size
      inputTensor.reshape([1, modelInputWidth, modelInputHeight, 3]),
    );

    setRawPredictions(prediction.arraySync()[0]);

    const choiceIndex = prediction.argMax(1).dataSync()[0];
    // $FlowFixMe
    return Object.values(choices)[choiceIndex];
  };

  const renderVideoSnapshot = (): ?HTMLCanvasElement => {
    if (!snapshotCanvasRef || !snapshotCanvasRef.current) {
      setErrorMessage('Canvas is not rendered');
      return null;
    }

    if (!videoFrame) {
      setErrorMessage('Cannot generate video snapshot');
      return null;
    }

    const canvas: HTMLCanvasElement = snapshotCanvasRef.current;
    const canvasContext: CanvasRenderingContext2D = canvas.getContext('2d');

    // Flip canvas if needed.
    if (!canvasFlipped && flipVideoHorizontally) {
      canvasContext.translate(canvas.width, 0);
      canvasContext.scale(-1, 1);
      setCanvasFlipped(true);
    }

    canvasContext.drawImage(
      videoFrame, 0, 0, canvas.width, canvas.height,
    );

    return canvas;
  };

  const clearVideoSnapshot = () => {
    if (!snapshotCanvasRef || !snapshotCanvasRef.current) {
      return;
    }
    const canvas: HTMLCanvasElement = snapshotCanvasRef.current;
    const canvasContext: CanvasRenderingContext2D = canvas.getContext('2d');
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  };

  const onVideoFrame = async (video?: ?HTMLVideoElement) => {
    setVideoFrame(video);
  };

  const onGameStart = () => {
    setCounter(countDownStart);
    setComputerChoice(null);
    setHumanChoice(null);
    setRawPredictions(null);
    clearVideoSnapshot();
  };

  const onGameEnd = () => {
    // Make a computer choice.
    const randomIndex: number = Math.floor(Math.random() * 3);
    // $FlowFixMe
    const computerRandomChoice: Choice = Object.values(choices)[randomIndex];
    setComputerChoice(computerRandomChoice);
    // Rendering a video snapshot.
    const currentCanvas: ?HTMLCanvasElement = renderVideoSnapshot();
    // Prediction user choice from the video snapshot.
    const humanChoicePrediction: ?Choice = predictHumanChoice(currentCanvas);
    setHumanChoice(humanChoicePrediction);
    // Detect the winner.
    if (!humanChoicePrediction) {
      setErrorMessage('Cannot predict human choice');
      return false;
    }

    if (
      humanChoicePrediction.beats.find(
        (choiceId: $Values<typeof choiceIDs>) => choiceId === computerRandomChoice.id,
      )
    ) {
      // Human won.
      setHumanScore(humanScore + 1);
    } else if (
      computerRandomChoice.beats.find(
        (choiceId: $Values<typeof choiceIDs>) => choiceId === humanChoicePrediction.id,
      )
    ) {
      // Computer won.
      setComputerScore(computerScore + 1);
    }

    return true;
  };

  const onGameEndCallback = useCallback(onGameEnd, [
    computerScore,
    computerChoice,
    humanScore,
    humanChoice,
    videoFrame,
    model,
    errorMessage,
    rawPredictions,
    counter,
  ]);

  // Countdown effect.
  useEffect(() => {
    if (counter === null) {
      // Nothing to count down.
      return;
    }
    if (counter === 0) {
      // Countdown is finished.
      setCounter(null);
      onGameEndCallback();
      return;
    }
    // Do the next tick.
    setTimeout(
      () => setCounter(counter - 1),
      countDownTimeout,
    );
  }, [onGameEndCallback, counter]);

  const cameraPaper = (
    <Paper className={classes.paper}>
      <CameraStream
        width={canvasWidth}
        height={canvasHeight}
        onVideoFrame={onVideoFrame}
        facingMode="user"
        flipHorizontal={flipVideoHorizontally}
      />
    </Paper>
  );

  const snapshotPaper = (
    <Paper className={classes.paperSnapshot}>
      <canvas
        width={canvasWidth}
        height={canvasHeight}
        ref={snapshotCanvasRef}
      />
    </Paper>
  );

  const humanChoiceBackgroundPaper = humanChoice ? (
    <Paper className={classes.paperHumanChoiceBackground} />
  ) : null;

  const humanChoicePaper = (
    <Paper className={classes.paperHumanChoice}>
      <Zoom in={!!humanChoice} timeout={{ enter: 200 }}>
        <Box className={classes.choice}>
          <span role="img" aria-label="Human Choice">
            {humanChoice && humanChoice.icon}
          </span>
        </Box>
      </Zoom>
    </Paper>
  );

  const computerChoicePaper = (
    <Paper className={classes.paper}>
      <Zoom in={!!computerChoice} timeout={{ enter: 200 }}>
        <Box className={classes.choice}>
          <span role="img" aria-label="Computer Choice">
            {computerChoice && computerChoice.icon}
          </span>
        </Box>
      </Zoom>
    </Paper>
  );

  const buttons = counter === null ? (
    <Box className={classes.buttons}>
      <Fab
        variant="extended"
        color="secondary"
        aria-label="Play"
        size="large"
        onClick={onGameStart}
      >
        <PlayArrowIcon />
        {' '}
        PLAY
      </Fab>
      {counter}
    </Box>
  ) : null;

  const score = (
    <Box className={classes.score}>
      {`${humanScore} : ${computerScore}`}
    </Box>
  );

  const countdown = counter !== null ? (
    <Box className={classes.countDown}>
      {counter}
    </Box>
  ) : null;

  const description = (
    <ol>
      <li>
        Press <b>PLAY</b>
      </li>
      <li>
        Wait for counter ticks: <b>3</b> → <b>2</b> → <b>1</b>
      </li>
      <li>
        Make you choice:
        {' '}
        {choices[choiceIDs.rock].icon} or {choices[choiceIDs.paper].icon}
        {' '}
        or {choices[choiceIDs.scissors].icon}
      </li>
      <li>Win the game!</li>
    </ol>
  );

  const oneHotPredictions: DataRecord[] = rawPredictions
    ? rawPredictions.map((value, index) => ({
      value,
      // $FlowFixMe
      label: Object.values(choices)[index].icon,
    }))
    : [];

  const oneHotBars = rawPredictions ? (
    <Box mt={3} mb={3} width={canvasWidth}>
      <Box mb={1}>
        <b>Probabilities</b>
      </Box>
      <OneHotBars data={oneHotPredictions} height={100} />
    </Box>
  ) : null;

  if (!model) {
    return (
      <Box>
        <Box>
          Loading the model
        </Box>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <>
      <Box className={classes.description}>
        {description}
      </Box>

      <Box>
        <Grid container spacing={4} alignItems="center" justify="flex-start">
          <Grid item>
            <Box className={classes.playerName}>
              <span role="img" aria-label="You">😎</span>
              {' '}
              You
            </Box>
            {cameraPaper}
            {snapshotPaper}
            {humanChoiceBackgroundPaper}
            {humanChoicePaper}
          </Grid>

          <Grid item>
            {score}
            {buttons}
            {countdown}
          </Grid>

          <Grid item>
            <Box className={classes.playerName}>
              <span role="img" aria-label="Computer">🤖</span>
              {' '}
              Computer
            </Box>
            {computerChoicePaper}
          </Grid>
        </Grid>
      </Box>

      <Box mb={3}>
        {oneHotBars}
      </Box>

      <Snack severity="error" message={errorMessage} />
    </>
  );
};

export default RockPaperScissors;
