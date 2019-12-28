import type { Node } from 'react';

export type ExperimentProps = {};

export type Experiment = {|
  slug: string,
  name: string,
  description: string,
  component: () => Node,
  colabURL: ?string,
  jupyterURL: ?string,
  cover: string,
|};

export type ExperimentsMap = {
  [string]: Experiment,
};
