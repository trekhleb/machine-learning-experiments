// @flow
import React from 'react';
import type { Node } from 'react';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import BugReportIcon from '@material-ui/icons/BugReport';
import GitHubIcon from '@material-ui/icons/GitHub';

import { ML_EXPERIMENTS_GITHUB_URL, ML_EXPERIMENTS_GITHUB_ISSUES_URL } from '../../constants/links';

const Footer = (): Node => {
  return (
    <>
      <Box display="flex" mt={2}>
        <Box mr={3}>
          <Button
            size="small"
            startIcon={<GitHubIcon />}
            onClick={() => window.open(ML_EXPERIMENTS_GITHUB_URL, '_blank')}
          >
            Contribute
          </Button>
        </Box>

        <Box>
          <Button
            size="small"
            startIcon={<BugReportIcon />}
            onClick={() => window.open(ML_EXPERIMENTS_GITHUB_ISSUES_URL, '_blank')}
          >
            Report an issue
          </Button>
        </Box>
      </Box>
    </>
  );
};

export default Footer;
