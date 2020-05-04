// @flow
import React from 'react';
import type { Node } from 'react';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import BugReportIcon from '@material-ui/icons/BugReport';
import GitHubIcon from '@material-ui/icons/GitHub';
import Tooltip from '@material-ui/core/Tooltip';

import { ML_EXPERIMENTS_GITHUB_URL, ML_EXPERIMENTS_GITHUB_ISSUES_URL } from '../../constants/links';

const Footer = (): Node => (
  <>
    <Box display="flex">
      <Box mr={3}>
        <Tooltip title="Machine Learning Experiments on GitHub">
          <Button
            size="small"
            startIcon={<GitHubIcon />}
            onClick={() => window.open(ML_EXPERIMENTS_GITHUB_URL, '_blank')}
          >
            Contribute
          </Button>
        </Tooltip>
      </Box>

      <Box>
        <Tooltip title="Create an issue on GitHub">
          <Button
            size="small"
            startIcon={<BugReportIcon />}
            onClick={() => window.open(ML_EXPERIMENTS_GITHUB_ISSUES_URL, '_blank')}
          >
            Report an issue
          </Button>
        </Tooltip>
      </Box>
    </Box>
  </>
);

export default Footer;
