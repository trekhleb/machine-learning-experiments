import React from 'react';
import type { Node } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import EmojiObjectsIcon from '@material-ui/icons/EmojiObjects';

const defaultProps = {
  imageWidth: 200,
};

type InputImagesExampleProps = {
  title: string,
  images: string[],
  imageWidth?: number,
};

const useStyles = makeStyles(() => ({
  heading: {
    display: 'flex',
    alignItems: 'center',
    fontWeight: 450,
  },
}));

const InputImagesExample = (props: InputImagesExampleProps): Node => {
  const { images, title, imageWidth = defaultProps.imageWidth } = props;

  const classes = useStyles();

  const imageList = images.map((imageSrc) => (
    <Grid item xs key={imageSrc}>
      <img src={imageSrc} alt="Input example" width={imageWidth} />
    </Grid>
  ));

  return (
    <Box>
      <ExpansionPanel>
        <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
          <Box className={classes.heading}>
            <EmojiObjectsIcon />
            {' '}
            {title}
          </Box>
        </ExpansionPanelSummary>
        <ExpansionPanelDetails>
          <Grid container spacing={1}>
            {imageList}
          </Grid>
        </ExpansionPanelDetails>
      </ExpansionPanel>
    </Box>
  );
};

InputImagesExample.defaultProps = defaultProps;

export default InputImagesExample;
