// @flow
import React from 'react';
import type { Node } from 'react';
import Button from '@material-ui/core/Button';
import Box from '@material-ui/core/Box';
import MenuBookIcon from '@material-ui/icons/MenuBook';
import GitHubIcon from '@material-ui/icons/GitHub';
import Tooltip from '@material-ui/core/Tooltip';

import { ML_EXPERIMENTS_GITHUB_URL, TREKHLEB_DEV_URL } from '../../constants/links';

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
        <Tooltip title="More articles about machine learning and web development">
          <Button
            size="small"
            startIcon={<MenuBookIcon />}
            onClick={() => window.open(TREKHLEB_DEV_URL, '_blank')}
          >
            trekhleb.dev
          </Button>
        </Tooltip>
      </Box>
    </Box>
  </>
);

export default Footer;
