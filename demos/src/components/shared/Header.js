import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { makeStyles } from '@material-ui/core/styles';
import MaterialLink from '@material-ui/core/Link';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
import Tooltip from '@material-ui/core/Tooltip';
import HomeIcon from '@material-ui/icons/Home';
import GitHubIcon from '@material-ui/icons/GitHub';

import { HOME_ROUTE } from '../../constants/routes';
import { MACHINE_LEARNING_EXPERIMENTS_GITHUB_URL } from '../../constants/links';

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

const Header = () => {
  const classes = useStyles();

  return (
    <>
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h6" className={classes.logoTypography} noWrap>
            <MaterialLink component={RouterLink} to={HOME_ROUTE} className={classes.logoLink}>
              Machine Learning Experiments
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
            <MaterialLink href={MACHINE_LEARNING_EXPERIMENTS_GITHUB_URL} color="inherit">
              <IconButton color="inherit">
                <GitHubIcon />
              </IconButton>
            </MaterialLink>
          </Tooltip>

        </Toolbar>
      </AppBar>
      <div className={classes.offset} />
    </>
  );
};

export default Header;
