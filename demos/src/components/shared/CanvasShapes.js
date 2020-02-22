// @flow
import React, { useRef, useEffect, useCallback } from 'react';
import type { Node } from 'react';

export type Box = {
  x: number,
  y: number,
  width: number,
  height: number,
  label: string,
};

type CanvasShapesProps = {
  boxes: ?Box[],
  canvasWidth: number,
  canvasHeight: number,
};

const boxColor = '#2fff00';
const boxFrameWidth = 1;
const boxLabelFont = '16px helvetica';
const boxLabelColor = '#000000';
const boxLabelPadding = 10;

const CanvasShapes = (props: CanvasShapesProps): Node => {
  const { boxes, canvasWidth, canvasHeight } = props;

  const canvasRef = useRef(null);

  const drawDetections = () => {
    if (!canvasRef.current) {
      return;
    }

    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = boxLabelFont;
    ctx.textBaseline = 'top';

    if (!boxes || !boxes.length) {
      return;
    }

    boxes.forEach((box: Box) => {
      const {
        x,
        y,
        width: detectionWidth,
        height: detectionHeight,
        label,
      } = box;

      // Draw the bounding box.
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = boxFrameWidth;
      ctx.strokeRect(x, y, detectionWidth, detectionHeight);

      // Draw the label background.
      ctx.fillStyle = boxColor;
      const textWidth = ctx.measureText(label).width;
      const textHeight = parseInt(boxLabelFont, 10);

      // Draw top left rectangle.
      ctx.fillRect(
        x,
        y,
        textWidth + boxLabelPadding,
        textHeight + boxLabelPadding,
      );

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = boxLabelColor;
      ctx.fillText(
        label,
        x + boxLabelPadding / 2,
        y + boxLabelPadding / 2,
      );
    });
  };

  const drawDetectionsCallback = useCallback(drawDetections, [boxes]);

  useEffect(() => {
    drawDetectionsCallback();
  }, [drawDetectionsCallback]);

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      style={canvasStyle}
    />
  );
};

const canvasStyle = {};

export default CanvasShapes;
