// @flow
import { GOOGLE_COLAB_URL, NB_VIEWER_URL } from '../constants/links';

// @see: https://mybinder.org/
export const generateJupyterLink = (notebookUrl: string): string => {
  const url = new URL(notebookUrl);
  return `${NB_VIEWER_URL}${url.pathname}`;
};

export const generateColabLink = (notebookUrl: string): string => {
  const url = new URL(notebookUrl);
  return `${GOOGLE_COLAB_URL}${url.pathname}`;
};
