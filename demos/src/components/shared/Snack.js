// @flow
import React, { useState, useEffect } from 'react';
import type { Node } from 'react';
import Snackbar from '@material-ui/core/Snackbar';
import Alert from '@material-ui/lab/Alert';
import type { Color } from '@material-ui/lab';

const autoHideDuration = 7000;

type SnackProps = {
  message: ?string,
  severity?: ?Color,
};

const defaultProps = {
  severity: 'success',
};

const Snack = (props: SnackProps): Node => {
  const { message, severity } = props;

  const [snackbarOpen, setSnackbarOpen] = useState(true);
  const [snackbarMessage, setSnackbarMessage] = useState(null);

  const onClose = () => {
    setSnackbarOpen(false);
  };

  useEffect(() => {
    if (message) {
      setSnackbarOpen(true);
    }
    setSnackbarMessage(message);
  }, [message]);

  if (!snackbarMessage) {
    return null;
  }

  return (
    <Snackbar
      open={snackbarOpen}
      autoHideDuration={autoHideDuration}
      onClose={onClose}
      anchorOrigin={{
        vertical: 'bottom',
        horizontal: 'center',
      }}
    >
      <Alert
        elevation={6}
        variant="filled"
        severity={severity || 'success'}
        onClose={onClose}
      >
        {snackbarMessage}
      </Alert>
    </Snackbar>
  );
};

Snack.defaultProps = defaultProps;

export default Snack;
