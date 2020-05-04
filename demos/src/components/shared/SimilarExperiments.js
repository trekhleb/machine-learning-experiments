// @flow
import React from 'react';
import type { Node } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import LibraryBooksIcon from '@material-ui/icons/LibraryBooks';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import { experimentsSlugs } from '../experiments/types';
import experiments from '../experiments';
import type { Experiment } from '../experiments/types';
import ExperimentPreview from './ExperimentPreview';

type InputImagesExampleProps = {
  similarExperiments: ?Array<$Values<typeof experimentsSlugs>>,
};

const useStyles = makeStyles(() => ({
  heading: {
    display: 'flex',
    alignItems: 'center',
    fontWeight: 450,
  },
}));

const SimilarExperiments = (props: InputImagesExampleProps): Node => {
  const { similarExperiments } = props;

  const classes = useStyles();

  if (!similarExperiments || !similarExperiments.length) {
    return null;
  }

  const experimentsList = similarExperiments.map(
    (experimentSlug: $Values<typeof experimentsSlugs>) => {
      if (!Object.prototype.hasOwnProperty.call(experiments, experimentSlug)) {
        return null;
      }
      const experiment: Experiment = experiments[experimentSlug];
      return (
        <Grid item key={experimentSlug} xs={12} sm={6} lg={3}>
          <ExperimentPreview experiment={experiment} />
        </Grid>
      );
    },
  );

  return (
    <ExpansionPanel>
      <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
        <Box className={classes.heading}>
          <Box mr={1}>
            <LibraryBooksIcon />
          </Box>
          <Box>
            Similar Experiments
          </Box>
        </Box>
      </ExpansionPanelSummary>
      <ExpansionPanelDetails>
        <Grid container spacing={1}>
          {experimentsList}
        </Grid>
      </ExpansionPanelDetails>
    </ExpansionPanel>
  );
};

export default SimilarExperiments;
