import React from 'react';
import {Link as RouterLink} from 'react-router-dom';
import MaterialLink from '@material-ui/core/Link';

import {HOME_ROUTE} from '../../constants/routes';

const Header = () => {
  return (
    <>
      <MaterialLink component={RouterLink} to={HOME_ROUTE}>
        Home
      </MaterialLink>
    </>
  );
};

export default Header;
