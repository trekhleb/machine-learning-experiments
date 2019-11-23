import React from 'react';
import {withRouter} from 'react-router-dom';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import CardMedia from '@material-ui/core/CardMedia';
import CardActions from '@material-ui/core/CardActions';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';

import {EXPERIMENTS_ROUTE} from '../../constants/routes';

const ExperimentPreview = (props) => {
  const {name, id, cover, history} = props;
  const experimentUrl = `${EXPERIMENTS_ROUTE}/${id}`;

  const onExperimentLaunch = () => {
    history.push(experimentUrl);
  };

  return (
    <Card>
      <CardMedia image={cover} title={name} style={cardMediaStyle} />
      <CardContent>
        <Typography variant="h5" component="h2">
          {name}
        </Typography>
      </CardContent>
      <CardActions>
        <Button
          variant="contained"
          color="default"
          startIcon={<PlayArrowIcon />}
          onClick={onExperimentLaunch}
        >
          Launch
        </Button>
      </CardActions>
    </Card>
  );
};

const cardMediaStyle = {
  height: '100px',
};


export default withRouter(ExperimentPreview);
