// @flow
import React from 'react';
import type { Node } from 'react';
import ExpansionPanel from '@material-ui/core/ExpansionPanel';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import Box from '@material-ui/core/Box';
import { makeStyles } from '@material-ui/core/styles';
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import InfoIcon from '@material-ui/icons/Info';

const useStyles = makeStyles(() => ({
  heading: {
    display: 'flex',
    alignItems: 'center',
    fontWeight: 450,
  },
}));

const InfoPanel = (): Node => {
  const classes = useStyles();

  return (
    <ExpansionPanel>
      <ExpansionPanelSummary expandIcon={<ExpandMoreIcon />}>
        <Box className={classes.heading}>
          <Box mr={1}>
            <InfoIcon />
          </Box>
          <Box>
            Attention
          </Box>
        </Box>
      </ExpansionPanelSummary>
      <ExpansionPanelDetails>
        <Box>
          {/* eslint-disable-next-line react/jsx-one-expression-per-line */}
          This web-site contains machine learning <b>experiments</b>. It is <b>not</b> a
          production ready, reusable, optimised and fine-tuned code and models.
          This is rather a sandbox or a playground for learning and trying different
          machine learning approaches, algorithms, data-sets.
        </Box>
      </ExpansionPanelDetails>
    </ExpansionPanel>
  );
};

export default InfoPanel;
