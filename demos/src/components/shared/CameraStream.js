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
const defaultFrameThrottling = 100;
const defaultFacingMode = 'environment';

const defaultProps = {
  onVideoFrame: (video: HTMLVideoElement) => {},
  maxWidth: defaultMaxWidth,
  maxHeight: defaultMaxHeight,
  videoFrameRate: defaultVideoFrameRate,
  frameThrottling: defaultFrameThrottling,
  facingMode: defaultFacingMode,
};

type CameraStreamProps = {
  width: number,
  height: number,
  maxWidth?: number,
  maxHeight?: number,
  facingMode?: string,
  videoFrameRate?: number,
  frameThrottling?: number,
  onVideoFrame?: (video: HTMLVideoElement) => void,
};

const CameraStream = (props: CameraStreamProps): Node => {
  const {
    width,
    height,
    onVideoFrame = (video) => {},
    maxWidth = defaultMaxWidth,
    maxHeight = defaultMaxHeight,
    facingMode,
    videoFrameRate,
    frameThrottling,
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
        onVideoFrameCallback(videoRef.current);
        throttledOnFrame();
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

    const userMediaConstraints = {
      video: {
        width: { ideal: width },
        height: { ideal: height },
        facingMode: { ideal: facingMode },
        frameRate: { ideal: videoFrameRate },
      },
      audio: false,
    };

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
        setErrorMessage('Video cannot be started');
      });

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

const videoStyle = {};

export default CameraStream;
