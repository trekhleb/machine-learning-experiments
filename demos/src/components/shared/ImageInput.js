import React from 'react';
import Box from '@material-ui/core/Box';

type ImageInputProps = {};

const ImageInput = (props: ImageInputProps) => {
  return (
    <Box>
      <input id="fileItem" type="file" />
    </Box>
  );
};

export default ImageInput;
