import React from 'react';
import {withRouter} from 'react-router-dom';
import {makeStyles} from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardActionArea from '@material-ui/core/CardActionArea';
import Typography from '@material-ui/core/Typography';
import CardMedia from '@material-ui/core/CardMedia';

import {EXPERIMENTS_ROUTE} from '../../constants/routes';

const ExperimentPreview = (props) => {
  const {name, id, cover, description, history} = props;
  const classes = useStyles(props);

  const experimentUrl = `${EXPERIMENTS_ROUTE}/${id}`;

  const onExperimentLaunch = () => {
    history.push(experimentUrl);
  };

  return (
    <Card>
      <CardActionArea onClick={onExperimentLaunch}>
        <CardMedia image={cover} title={name} className={classes.media} />
        <CardContent>
          <Typography gutterBottom variant="h5" component="h2">
            {name}
          </Typography>
          <Typography variant="body2" color="textSecondary" component="p">
            {description}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
};

const useStyles = makeStyles(theme => ({
  media: {
    height: 200,
  },
}));

export default withRouter(ExperimentPreview);
