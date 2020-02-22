// @flow
import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { Node } from 'react';

const defaultBackgroundColor = '#FFFFFF';

const defaultProps = {
  width: 200,
  height: 200,
  lineColor: '#000000',
  lineWidth: 16,
  lineJoin: 'round',
  revision: 0,
  backgroundColor: defaultBackgroundColor,
  onDrawEnd: () => {},
};

export type CanvasImages = {
  imageData: ?ImageData,
};

type Coordinate = {
  x: number,
  y: number,
};

type CanvasProps = {
  width?: number,
  height?: number,
  lineWidth?: number,
  lineJoin?: string,
  lineColor?: string,
  backgroundColor?: string,
  revision?: number,
  onDrawEnd?: (canvasImages: CanvasImages) => void,
};

// @see: https://dev.to/ankursheel/react-component-to-fraw-on-a-page-using-hooks-and-typescript-2ahp
const Canvas = (props: CanvasProps): Node => {
  const {
    width,
    height,
    lineColor,
    lineWidth,
    lineJoin,
    backgroundColor = defaultBackgroundColor,
    revision,
    onDrawEnd: onDrawEndCallback = (canvasImages: CanvasImages) => canvasImages,
  } = props;

  const canvasRef = useRef(null);
  const [isPainting, setIsPainting] = useState(false);
  const [mousePosition, setMousePosition] = useState(undefined);

  const getCoordinates = (event: MouseEvent): ?Coordinate => {
    if (!canvasRef.current) {
      return null;
    }
    const canvas: HTMLCanvasElement = canvasRef.current;
    return {
      x: event.pageX - canvas.offsetLeft,
      y: event.pageY - canvas.offsetTop,
    };
  };

  const startPaint = useCallback((event: MouseEvent) => {
    const coordinates = getCoordinates(event);
    if (coordinates) {
      setMousePosition(coordinates);
      setIsPainting(true);
    }
  }, []);

  const paint = useCallback(
    (event: MouseEvent) => {
      const drawLine = (originalMousePosition: Coordinate, newMousePosition: Coordinate) => {
        if (!canvasRef.current) {
          return;
        }
        const canvas: HTMLCanvasElement = canvasRef.current;
        const context = canvas.getContext('2d');
        if (context) {
          context.strokeStyle = lineColor || defaultProps.lineColor;
          context.lineJoin = lineJoin || defaultProps.lineJoin;
          context.lineWidth = lineWidth || defaultProps.lineWidth;
          context.beginPath();
          context.moveTo(originalMousePosition.x, originalMousePosition.y);
          context.lineTo(newMousePosition.x, newMousePosition.y);
          context.closePath();
          context.stroke();
        }
      };

      if (isPainting) {
        const newMousePosition = getCoordinates(event);
        if (mousePosition && newMousePosition) {
          drawLine(mousePosition, newMousePosition);
          setMousePosition(newMousePosition);
        }
      }
    },
    [isPainting, mousePosition, lineColor, lineJoin, lineWidth],
  );

  const exitPaint = useCallback(() => {
    const onDrawEnd = () => {
      if (!canvasRef.current) {
        return;
      }

      const canvas: HTMLCanvasElement = canvasRef.current;
      const context = canvas.getContext('2d');

      // Call a callback.
      onDrawEndCallback({
        imageData: context.getImageData(0, 0, canvas.width, canvas.height),
      });
    };

    onDrawEnd();
    setIsPainting(false);
    setMousePosition(undefined);
  }, [onDrawEndCallback]);

  // Effect for MouseDown.
  useEffect(() => {
    if (!canvasRef.current) {
      return () => {};
    }
    const canvas: HTMLCanvasElement = canvasRef.current;
    canvas.addEventListener('mousedown', startPaint);
    // $FlowFixMe
    canvas.addEventListener('touchstart', startPaint);

    return () => {
      canvas.removeEventListener('mousedown', startPaint);
      // $FlowFixMe
      canvas.removeEventListener('touchstart', startPaint);
    };
  }, [startPaint]);

  // Effect for MouseMove.
  useEffect(() => {
    if (!canvasRef.current) {
      return () => {};
    }
    const canvas: HTMLCanvasElement = canvasRef.current;
    canvas.addEventListener('mousemove', paint);
    // $FlowFixMe
    canvas.addEventListener('touchmove', paint);

    return () => {
      canvas.removeEventListener('mousemove', paint);
      // $FlowFixMe
      canvas.removeEventListener('touchmove', paint);
    };
  }, [paint]);

  // Effect for MouseUp and MouseLeave.
  useEffect(() => {
    if (!canvasRef.current) {
      return () => {};
    }
    const canvas: HTMLCanvasElement = canvasRef.current;
    canvas.addEventListener('mouseup', exitPaint);
    canvas.addEventListener('mouseleave', exitPaint);
    canvas.addEventListener('touchend', exitPaint);
    canvas.addEventListener('touchcancel', exitPaint);

    return () => {
      canvas.removeEventListener('mouseup', exitPaint);
      canvas.removeEventListener('mouseleave', exitPaint);
      canvas.removeEventListener('touchend', exitPaint);
      canvas.removeEventListener('touchcancel', exitPaint);
    };
  }, [exitPaint]);

  // Effect for filling a canvases with color.
  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }
    const canvas: HTMLCanvasElement = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, [backgroundColor]);

  // Effect to cleanup the canvas.
  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }
    const canvas: HTMLCanvasElement = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, [revision, backgroundColor]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ touchAction: 'none' }}
    />
  );
};

Canvas.defaultProps = defaultProps;

export default Canvas;
