export const imageDataToGreyScaleMatrix = (imageData: ImageData) => {
  const {data, width, height} = imageData;
  const imageDataLayersNum = 4; // RGBA
  const matrix = [];

  if (data.length !== (imageDataLayersNum * width * height)) {
    throw new Error(
      `Incorrect imageData format: sizes do not match (${data.length} !== ${width} x ${height})`
    );
  }

  for (let row = 0; row < height; row += 1) {
    matrix[row] = [];
    for (let col = 0; col < width; col += 1) {
      const shift = (row * width + col) * imageDataLayersNum;
      const r = data[shift];
      const g = data[shift + 1];
      const b = data[shift + 2];
      const a = data[shift + 3];
      matrix[row][col] = rgbToGreyScale(r, g, b);
    }
  }

  return matrix;
};

// r, g, b - are in [0..255] range.
// @see: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
const rgbToGreyScale = (r: number, g: number, b: number) => {
  const cLinear = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  const cSRGB = cLinear < 0.0031308 ? (12.92 * cLinear) : (1.055 * (cLinear ** (1/2.4)) - 0.055);
  return Math.round(cSRGB * 255);
};
