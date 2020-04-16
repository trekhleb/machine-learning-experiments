// @flow
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
  title: 'Input text examples (for better model performance)',
};

type InputTextsExampleProps = {
  texts: ?string[],
  title?: string,
};

const useStyles = makeStyles(() => ({
  heading: {
    display: 'flex',
    alignItems: 'center',
    fontWeight: 450,
  },
}));

const InputTextsExample = (props: InputTextsExampleProps): Node => {
  const {
    texts,
    title = defaultProps.title,
  } = props;

  const classes = useStyles();

  if (!texts) {
    return null;
  }

  const textsList = texts.map((text: string) => (
    <li key={text}>
      <i>{`"${text}"`}</i>
    </li>
  ));

  return (
    <ExpansionPanel>
      <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
        <Box className={classes.heading}>
          <Box mr={1}>
            <EmojiObjectsIcon />
          </Box>
          <Box>
            {title}
          </Box>
        </Box>
      </ExpansionPanelSummary>
      <ExpansionPanelDetails>
        <Grid container spacing={1}>
          <ul>
            {textsList}
          </ul>
        </Grid>
      </ExpansionPanelDetails>
    </ExpansionPanel>
  );
};

InputTextsExample.defaultProps = defaultProps;

export default InputTextsExample;
