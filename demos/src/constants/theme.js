// @flow
import { createMuiTheme } from '@material-ui/core/styles';
// import {red} from '@material-ui/core/colors';

// @see: https://github.com/mui-org/material-ui/blob/master/examples/create-react-app/src/theme.js
const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#24292e',
    },
    // secondary: {
    //   main: '#19857b',
    // },
    // error: {
    //   main: red.A400,
    // },
    // background: {
    //   default: '#fff',
    // },
  },
});

export default theme;
