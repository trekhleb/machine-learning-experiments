// @flow
import React from 'react';
import type { Node } from 'react';
import { withRouter } from 'react-router-dom';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardActionArea from '@material-ui/core/CardActionArea';
import Typography from '@material-ui/core/Typography';
import CardMedia from '@material-ui/core/CardMedia';
import type { RouterHistory } from 'react-router-dom';

import { EXPERIMENTS_ROUTE } from '../../constants/routes';
import type { Experiment } from '../experiments/types';

const useStyles = makeStyles(() => ({
  media: {
    height: 200,
  },
}));

type ExperimentPreviewProps = {
  experiment: Experiment,
  history: RouterHistory,
};

const ExperimentPreview = (props: ExperimentPreviewProps): Node => {
  const { experiment, history } = props;
  const classes = useStyles(props);

  const experimentUrl = `${EXPERIMENTS_ROUTE}/${experiment.slug}`;

  const onExperimentLaunch = () => {
    history.push(experimentUrl);
  };

  return (
    <Card>
      <CardActionArea onClick={onExperimentLaunch}>
        <CardMedia image={experiment.cover} title={experiment.name} className={classes.media} />
        <CardContent>
          <Typography gutterBottom variant="h6" component="h2">
            {experiment.name}
          </Typography>
          <Typography variant="body2" color="textSecondary" component="p">
            {experiment.description}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
};

export default withRouter(ExperimentPreview);
