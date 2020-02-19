import React, { useState, useEffect, useRef } from 'react';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import Grid from '@material-ui/core/Grid';
import Fab from '@material-ui/core/Fab';
import Zoom from '@material-ui/core/Zoom';

import cover from './cover.jpg';
import { ML_EXPERIMENTS_DEMO_MODELS_PATH, ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL } from '../../../constants/links';
import type { Experiment } from '../types';
import CameraStream from '../../shared/CameraStream';

const experimentSlug = 'RockPaperScissorsCNN';
const experimentName = 'Rock Paper Scissors (CNN)';
const experimentDescription = 'Play Rock Paper Scissors game against computer using Convolutional Neural Network (CNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/rock_paper_scissors_cnn/model.json`;

const gameStates = {
  notStarted: 'notStarted',
  inProgress: 'inProgress',
  predicting: 'predicting',
  finished: 'finished',
};

type GameState = $Values<typeof gameStates>;

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

const RockPaperScissorsCNN = (): Node => {
  const [computerChoice, setComputerChoice] = useState(null);
  const [humanChoice, setHumanChoice] = useState(null);
  const [humanScore, setHumanScore] = useState(0);
  const [computerScore, setComputerScore] = useState(0);
  const [counter, setCounter] = useState(null);
  const [videoFrame, setVideoFrame] = useState(null);
  const [gameState, setGameState] = useState(gameStates.notStarted);

  const snapshotCanvasRef = useRef(null);

  const classes = useStyles();

  const renderVideoSnapshot = () => {
    if (!snapshotCanvasRef || !snapshotCanvasRef.current) {
      return;
    }
    const canvas: HTMLCanvasElement = snapshotCanvasRef.current;
    const canvasContext: CanvasRenderingContext2D = canvas.getContext('2d');
    canvasContext.drawImage(
      videoFrame, 0, 0, canvas.width, canvas.height,
    );
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
    setGameState(gameStates.inProgress);
    clearVideoSnapshot();
  };

  const onGameEnd = () => {
    const randomIndex: number = Math.floor(Math.random() * 3);
    const computerRandomChoice: Choice = Object.values(choices)[randomIndex];
    setComputerChoice(computerRandomChoice);
    setGameState(gameStates.predicting);
    renderVideoSnapshot();
  };

  // Flip canvas horizontally if needed.
  useEffect(() => {
    if (!snapshotCanvasRef || !snapshotCanvasRef.current) {
      return;
    }
    if (!flipVideoHorizontally) {
      return;
    }
    const canvas: HTMLCanvasElement = snapshotCanvasRef.current;
    const canvasContext: CanvasRenderingContext2D = canvas.getContext('2d');
    canvasContext.translate(canvas.width, 0);
    canvasContext.scale(-1, 1);
  }, []);

  // Countdown effect.
  useEffect(() => {
    if (counter === null) {
      // Nothing to count down.
      return;
    }
    if (counter === 0) {
      // Countdown is finished.
      setCounter(null);
      onGameEnd();
      return;
    }
    // Do the next tick.
    setTimeout(
      () => setCounter(counter - 1),
      countDownTimeout,
    );
  }, [counter]);

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

  const computerChoicePaper = (
    <Paper className={classes.paper}>
      <Zoom in={!!computerChoice} timeout={{ enter: 200 }}>
        <Box className={classes.choice}>
          <span role="img" aria-label="Choice">
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
        Press
        {' '}
        <b>PLAY</b>
      </li>
      <li>
        Wait for counter ticks:
        {' '}
        <b>3</b>
        {' '}
        →
        {' '}
        <b>2</b>
        {' '}
        →
        {' '}
        <b>1</b>
      </li>
      <li>
        Make you choice:
        {' '}
        {choices[choiceIDs.rock].icon}
        {' '}
        or
        {' '}
        {choices[choiceIDs.paper].icon}
        {' '}
        or
        {' '}
        {choices[choiceIDs.scissors].icon}
      </li>
      <li>Win the game!</li>
    </ol>
  );

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
    </>
  );
};

const experiment: Experiment = {
  slug: experimentSlug,
  name: experimentName,
  description: experimentDescription,
  component: RockPaperScissorsCNN,
  notebookUrl,
  cover,
};

export default experiment;
