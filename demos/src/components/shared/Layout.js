import React from 'react';
import Grid from '@material-ui/core/Grid';
import Container from '@material-ui/core/Container';
import CssBaseline from '@material-ui/core/CssBaseline';
import 'typeface-roboto';

import PrimaryMenu from './PrimaryMenu';

const RootLayout = (props) => {
  const {children} = props;

  return (
    <>
      <CssBaseline />
      <Container maxWidth="xl">
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <PrimaryMenu />
          </Grid>
          <Grid item xs={12}>
            {children}
          </Grid>
        </Grid>
      </Container>
    </>
  );
};

export default RootLayout;
