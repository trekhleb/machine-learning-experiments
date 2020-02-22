// @flow
import React from 'react';
import ErrorIcon from '@material-ui/icons/Error';
import Box from '@material-ui/core/Box';

class ErrorBoundary extends React.Component<any, any> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: any) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    /* eslint-disable-next-line no-console */
    console.error(error, errorInfo);
  }

  render() {
    const { hasError } = this.state;
    const { children } = this.props;

    if (hasError) {
      return (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          m={2}
        >
          <ErrorIcon />
          {' '}
          Component has crashed
        </Box>
      );
    }

    return children;
  }
}

export default ErrorBoundary;
