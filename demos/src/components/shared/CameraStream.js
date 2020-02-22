// @flow
import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import throttle from 'lodash/throttle';
import type { Node } from 'react';
import Snack from './Snack';

const defaultMaxWidth = 500;
const defaultMaxHeight = 500;
const defaultVideoFrameRate = 30;
const defaultFrameThrottling = 200;
const defaultFacingMode = 'environment';

const defaultProps = {
  onVideoFrame: (video: HTMLVideoElement) => {},
  maxWidth: defaultMaxWidth,
  maxHeight: defaultMaxHeight,
  videoFrameRate: defaultVideoFrameRate,
  frameThrottling: defaultFrameThrottling,
  facingMode: defaultFacingMode,
  flipHorizontal: false,
};

type CameraStreamProps = {
  width: number,
  height: number,
  maxWidth?: number,
  maxHeight?: number,
  facingMode?: string,
  videoFrameRate?: number,
  frameThrottling?: number,
  onVideoFrame?: (video?: ?HTMLVideoElement) => Promise<void>,
  flipHorizontal?: boolean,
};

const CameraStream = (props: CameraStreamProps): Node => {
  const {
    width,
    height,
    onVideoFrame = async (video) => {},
    maxWidth = defaultMaxWidth,
    maxHeight = defaultMaxHeight,
    facingMode,
    videoFrameRate,
    frameThrottling,
    flipHorizontal,
  } = props;

  const videoRef = useRef(null);

  const [errorMessage, setErrorMessage] = useState(null);

  const onVideoFrameCallback = useCallback(onVideoFrame, []);

  useEffect(() => {
    if (!videoRef.current) {
      return () => {};
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage('Camera access is not supported by your browser');
      return () => {};
    }

    let localStream = null;
    let localAnimationRequestID = null;

    const onFrame = () => {
      localAnimationRequestID = requestAnimationFrame(() => {
        onVideoFrameCallback(videoRef.current).then(() => {
          throttledOnFrame();
        });
      });
    };

    const throttledOnFrame = throttle(
      onFrame,
      frameThrottling,
      {
        leading: false,
        trailing: true,
      },
    );

    const userMediaConstraints: MediaStreamConstraints = {
      video: {
        width: { ideal: width },
        height: { ideal: height },
        facingMode: { ideal: facingMode },
        frameRate: { ideal: videoFrameRate },
      },
      audio: false,
    };

    if (navigator.mediaDevices) {
      navigator.mediaDevices
        .getUserMedia(userMediaConstraints)
        .then((stream: MediaStream) => {
          if (!videoRef.current) {
            return;
          }
          localStream = stream;
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            throttledOnFrame();
          };
        })
        .catch((e) => {
          let message = 'Video cannot be started';
          if (e && e.message) {
            message += `: ${e.message}`;
          }
          setErrorMessage(message);
        });
    }

    return () => {
      // Stop animation frames.
      throttledOnFrame.cancel();
      if (localAnimationRequestID) {
        cancelAnimationFrame(localAnimationRequestID);
      }
      // Stop camera access.
      if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [
    width,
    height,
    facingMode,
    onVideoFrameCallback,
    videoFrameRate,
    frameThrottling,
  ]);

  const videoStyle = {
    transform: flipHorizontal ? 'scaleX(-1)' : '',
  };

  return (
    <>
      <video
        playsInline
        autoPlay
        muted
        ref={videoRef}
        width={Math.min(width, maxWidth)}
        height={Math.min(height, maxHeight)}
        style={videoStyle}
      />
      <Snack severity="error" message={errorMessage} />
    </>
  );
};

CameraStream.defaultProps = defaultProps;

export default CameraStream;
