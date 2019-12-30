import React from 'react';
import type { Node } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { makeStyles } from '@material-ui/core/styles';
import MaterialLink from '@material-ui/core/Link';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import EmojiObjectsIcon from '@material-ui/icons/EmojiObjects';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
import Hidden from '@material-ui/core/Hidden';
import Tooltip from '@material-ui/core/Tooltip';
import HomeIcon from '@material-ui/icons/Home';
import Slide from '@material-ui/core/Slide';
import useScrollTrigger from '@material-ui/core/useScrollTrigger';
import GitHubIcon from '@material-ui/icons/GitHub';

import { HOME_ROUTE } from '../../constants/routes';
import { ML_EXPERIMENTS_GITHUB_URL } from '../../constants/links';

const useStyles = makeStyles((theme) => ({
  logoTypography: {
    flexGrow: 1,
  },
  logoLink: {
    color: 'inherit',
    '&:hover': {
      textDecoration: 'none',
    },
  },
  offset: theme.mixins.toolbar,
}));

const Header = (): Node => {
  const classes = useStyles();
  const scrollTrigger = useScrollTrigger();

  return (
    <>
      <Slide appear={false} direction="down" in={!scrollTrigger}>
        <AppBar position="fixed">
          <Toolbar>
            <Typography variant="h6" className={classes.logoTypography} noWrap>
              <MaterialLink component={RouterLink} to={HOME_ROUTE} className={classes.logoLink}>
                <EmojiObjectsIcon style={{ marginBottom: '-4px' }} />
                {' '}
                <Hidden only={['xs']}>
                  Machine Learning Experiments
                </Hidden>
                <Hidden only={['sm', 'md', 'lg', 'xl']}>
                  ML Experiments
                </Hidden>
              </MaterialLink>
            </Typography>

            <Tooltip title="ML Experiments List">
              <MaterialLink component={RouterLink} to={HOME_ROUTE} color="inherit">
                <IconButton color="inherit">
                  <HomeIcon />
                </IconButton>
              </MaterialLink>
            </Tooltip>

            <Tooltip title="ML Experiments on GitHub">
              <MaterialLink href={ML_EXPERIMENTS_GITHUB_URL} color="inherit">
                <IconButton color="inherit">
                  <GitHubIcon />
                </IconButton>
              </MaterialLink>
            </Tooltip>
          </Toolbar>
        </AppBar>
      </Slide>
      <div className={classes.offset} />
    </>
  );
};

export default Header;
