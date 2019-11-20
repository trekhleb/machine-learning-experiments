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
  const {name, id, previewUrl, history} = props;
  const experimentUrl = `${EXPERIMENTS_ROUTE}/${id}`;

  const onTry = () => {
    history.push(experimentUrl);
  };

  return (
    <Card>
      {previewUrl && <CardMedia image={previewUrl} title={name} />}
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
          onClick={onTry}
        >
          Try
        </Button>
      </CardActions>
    </Card>
  );
};

export default withRouter(ExperimentPreview);
