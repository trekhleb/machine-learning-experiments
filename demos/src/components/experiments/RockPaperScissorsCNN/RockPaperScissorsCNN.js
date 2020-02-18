import React, { useState, useEffect } from 'react';
import type { Node } from 'react';
import Paper from '@material-ui/core/Paper';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import Grid from '@material-ui/core/Grid';
import Fab from '@material-ui/core/Fab';
import Looks3Icon from '@material-ui/icons/Looks3';
import LooksTwoIcon from '@material-ui/icons/LooksTwo';
import LooksOneIcon from '@material-ui/icons/LooksOne';

import cover from './cover.jpg';
import { ML_EXPERIMENTS_DEMO_MODELS_PATH, ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL } from '../../../constants/links';
import type { Experiment } from '../types';
import CameraStream from '../../shared/CameraStream';

const experimentSlug = 'RockPaperScissorsCNN';
const experimentName = 'Rock Paper Scissors (CNN)';
const experimentDescription = 'Play Rock Paper Scissors game against computer using Convolutional Neural Network (CNN)';
const notebookUrl = `${ML_EXPERIMENTS_GITHUB_NOTEBOOKS_URL}/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb`;

const modelPath = `${ML_EXPERIMENTS_DEMO_MODELS_PATH}/rock_paper_scissors_cnn/model.json`;

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
  },
  playerName: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    fontWeight: 450,
    marginBottom: '10px',
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
  const [computerChoice, setComputerChoice] = useState('');

  const classes = useStyles();

  const onVideoFrame = async (video?: ?HTMLVideoElement) => {
    console.log('+++ On video frame');
  };

  const onPlay = () => {
    console.log('+++ PLAY');
  };

  const cameraPaper = (
    <>
      <Box className={classes.playerName}>
        <span role="img" aria-label="You">😎</span>
        {' '}
        You
      </Box>
      <Paper className={classes.paper}>
        <CameraStream
          width={canvasWidth}
          height={canvasHeight}
          onVideoFrame={onVideoFrame}
          facingMode="user"
          flipHorizontal
        />
      </Paper>
    </>
  );

  const computerChoicePaper = (
    <>
      <Box className={classes.playerName}>
        <span role="img" aria-label="Computer">🤖</span>
        {' '}
        Computer
      </Box>
      <Paper className={classes.paper}>
        <Box className={classes.choice}>
          <span role="img" aria-label="Choice">
            {computerChoice && computerChoice.icon}
          </span>
        </Box>
      </Paper>
    </>
  );

  const buttons = (
    <Box className={classes.buttons}>
      <Fab
        variant="extended"
        color="secondary"
        aria-label="Play"
        size="large"
        onClick={onPlay}
      >
        <PlayArrowIcon />
        {' '}
        PLAY
      </Fab>
    </Box>
  );

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
            {cameraPaper}
          </Grid>

          <Grid item>
            {buttons}
          </Grid>

          <Grid item>
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
