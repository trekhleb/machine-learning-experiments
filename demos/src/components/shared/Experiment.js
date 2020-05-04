// @flow
import React from 'react';
import type { Node } from 'react';
import { withRouter } from 'react-router-dom';
import type { Match } from 'react-router-dom';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import { Helmet } from 'react-helmet';
import ErrorIcon from '@material-ui/icons/Error';
import Typography from '@material-ui/core/Typography';

import type { Experiment as ExperimentType } from '../experiments/types';
import experiments from '../experiments';
import { EXPERIMENT_ID_PARAM } from '../../constants/routes';
import { ML_EXPERIMENTS_GITHUB_URL } from '../../constants/links';
import Badge, { badgeType } from './Badge';
import { WINDOW_TITLE } from '../../constants/copies';
import { generateColabLink, generateJupyterLink } from '../../utils/links';
import ErrorBoundary from './ErrorBoundary';
import InfoPanel from './InfoPanel';
import InputImagesExample from './InputImagesExample';
import InputTextsExample from './InputTextsExample';
import SimilarExperiments from './SimilarExperiments';

type ExperimentProps = {
  match: Match,
};

const Experiment = (props: ExperimentProps): Node => {
  const { match } = props;

  const experimentId: ?string = match.params[EXPERIMENT_ID_PARAM];

  const unknownExperiment = (
    <Box display="flex" justifyContent="flex-start">
      <ErrorIcon fontSize="small" />
      <Box ml={1}>Unknown experiment</Box>
    </Box>
  );

  if (!experimentId) {
    return unknownExperiment;
  }

  if (!Object.prototype.hasOwnProperty.call(experiments, experimentId)) {
    return unknownExperiment;
  }

  const experiment: ExperimentType = experiments[experimentId];
  const ExperimentElement = experiment.component;

  const githubLink = (
    <Grid item>
      <Badge
        url={ML_EXPERIMENTS_GITHUB_URL}
        type={badgeType.github}
      />
    </Grid>
  );

  const jupyterLink = experiment.notebookUrl ? (
    <Grid item>
      <Badge
        url={generateJupyterLink(experiment.notebookUrl)}
        type={badgeType.jupyter}
      />
    </Grid>
  ) : null;

  const colabLink = experiment.notebookUrl ? (
    <Grid item>
      <Badge
        url={generateColabLink(experiment.notebookUrl)}
        type={badgeType.colab}
      />
    </Grid>
  ) : null;

  const inputImageExamples = experiment.inputImageExamples ? (
    <InputImagesExample
      imageWidth={experiment.inputImageExamples.imageWidth}
      images={experiment.inputImageExamples.images}
    />
  ) : null;

  const inputTextExamples = experiment.inputTextExamples ? (
    <InputTextsExample texts={experiment.inputTextExamples} />
  ) : null;

  return (
    <ErrorBoundary>
      <Helmet>
        <title>{`${WINDOW_TITLE} | ${experiment.name}`}</title>
        <meta name="description" content={experiment.description} />
      </Helmet>
      <Box mb={1}>
        <Typography variant="h5">
          {experiment.name}
        </Typography>
      </Box>
      <Box mb={1}>
        <Typography variant="body1" component="p">
          {experiment.description}
        </Typography>
      </Box>
      <Box mb={3}>
        <Grid container spacing={1} alignItems="center" justify="flex-start">
          {jupyterLink}
          {colabLink}
          {githubLink}
        </Grid>
      </Box>
      <Box mb={6}>
        <ExperimentElement />
      </Box>
      <Box>
        {inputImageExamples}
        {inputTextExamples}
        <SimilarExperiments similarExperiments={experiment.similarExperiments || null} />
        <InfoPanel />
      </Box>
    </ErrorBoundary>
  );
};

export default withRouter(Experiment);
