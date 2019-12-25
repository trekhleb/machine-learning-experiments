import type { Node } from 'react';

export type ExperimentProps = {};

export type Experiment = {|
  slug: string,
  name: string,
  description: string,
  component: () => Node,
  trainingURL: string,
  cover: string,
|};

export type ExperimentsMap = {
  [string]: Experiment,
};
