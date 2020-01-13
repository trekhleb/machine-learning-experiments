import React, { useRef, useEffect, useCallback } from 'react';
import type { Node } from 'react';

export type Detection = {
  leftTopX: number,
  leftTopY: number,
  width: number,
  height: number,
  label: number,
  probability: number,
};

type ObjectsDetectionsProps = {
  detections: ?Detection[],
  width: number,
  height: number,
};

const detectionColor = '#2fff00';
const detectionFrameWidth = 1;
const detectionLabelFont = '24px helvetica';
const detectionLabelColor = '#000000';

const ObjectsDetections = (props: ObjectsDetectionsProps): Node => {
  const { detections, width, height } = props;

  const canvasRef = useRef(null);

  const drawDetections = () => {
    if (!canvasRef.current) {
      return;
    }

    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.font = detectionLabelFont;
    ctx.textBaseline = 'top';

    if (!detections || !detections.length) {
      return;
    }

    detections.forEach((detection: Detection) => {
      const {
        leftTopX,
        leftTopY,
        width: detectionWidth,
        height: detectionHeight,
        label,
        probability,
      } = detection;

      // Draw the bounding box.
      ctx.strokeStyle = detectionColor;
      ctx.lineWidth = detectionFrameWidth;
      ctx.strokeRect(leftTopX, leftTopY, detectionWidth, detectionHeight);

      // Draw the label background.
      ctx.fillStyle = detectionColor;
      const textWidth = ctx.measureText(label).width;
      const textHeight = parseInt(detectionLabelFont, 10);

      // Draw top left rectangle.
      ctx.fillRect(
        leftTopX,
        leftTopY,
        textWidth + 10,
        textHeight + 10,
      );

      // Draw bottom left rectangle.
      ctx.fillRect(
        leftTopX,
        leftTopY + detectionHeight - textHeight,
        textWidth + 15,
        textHeight + 10,
      );

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = detectionLabelColor;
      ctx.fillText(label, leftTopX, leftTopY);
      ctx.fillText(
        probability.toFixed(2),
        leftTopX,
        leftTopY + detectionHeight - textHeight,
      );
    });
  };

  const drawDetectionsCallback = useCallback(drawDetections, [detections]);

  useEffect(() => {
    drawDetectionsCallback();
  }, [drawDetectionsCallback]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={canvasStyle}
    />
  );
};

const canvasStyle = {
  background: 'yellow',
};

export default ObjectsDetections;
