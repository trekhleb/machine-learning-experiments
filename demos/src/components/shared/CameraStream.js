import React, { useEffect, useRef, useState } from 'react';
import type { Node } from 'react';
import Snack from './Snack';

type CameraStreamProps = {
  width: number,
  height: number,
};

const CameraStream = (props: CameraStreamProps): Node => {
  const { width, height } = props;

  const videoRef = useRef(null);

  const [errorMessage, setErrorMessage] = useState(null);

  // Request the access to camera.
  useEffect(() => {
    if (!videoRef.current) {
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage('Camera access is not supported by your browser');
      return;
    }

    const userMediaConstraints = {
      video: {
        width: { ideal: width },
        height: { ideal: height },
        facingMode: { ideal: 'environment' },
      },
      audio: false,
    };

    navigator.mediaDevices
      .getUserMedia(userMediaConstraints)
      .then((stream: MediaStream) => {
        if (!videoRef.current) {
          return;
        }
        videoRef.current.srcObject = stream;
        // videoRef.current.onloadedmetadata = () => {
        //   onVideoFrameCallback();
        // };
      })
      .catch((e) => {
        setErrorMessage('Video cannot be started');
      });
  }, [width, height]);

  return (
    <>
      <video
        playsInline
        autoPlay
        muted
        ref={videoRef}
        width={width}
        height={height}
        style={videoStyle}
      />
      <Snack severity="error" message={errorMessage} />
    </>
  );
};

const videoStyle = {};

export default CameraStream;
