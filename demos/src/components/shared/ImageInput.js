import React, { useRef } from 'react';
import ImageIcon from '@material-ui/icons/Image';
import Button from '@material-ui/core/Button';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(() => ({
  imageInput: {
    display: 'none',
  },
}));

type ImageInputProps = {
  onSelect: (images: File[]) => void,
};

const ImageInput = (props: ImageInputProps) => {
  const { onSelect } = props;

  const classes = useStyles();

  const fileInputRef = useRef(null);

  const onFileSelect = (event) => {
    const imagesList: FileList = event.target.files;
    const imagesArray = Array.from(imagesList);
    onSelect(imagesArray);
  };

  const onFileButtonClick = () => {
    if (!fileInputRef.current) {
      return;
    }
    fileInputRef.current.click();
  };

  return (
    <>
      <input
        type="file"
        accept="image/*"
        multiple={false}
        onChange={onFileSelect}
        className={classes.imageInput}
        ref={fileInputRef}
      />
      <Button
        startIcon={<ImageIcon />}
        variant="contained"
        color="primary"
        onClick={onFileButtonClick}
      >
        Choose image
      </Button>
    </>
  );
};

export default ImageInput;
