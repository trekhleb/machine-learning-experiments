export type ExperimentProps = {};

export type Experiment = {
  name: string,
  description: string,
  component: Function,
  cover: string,
};

export type ExperimentsMap = {
  [string]: Experiment,
};
