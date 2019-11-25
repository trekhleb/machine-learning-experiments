export type ExperimentProps = {};

export type Experiment = {|
  slug: string,
  name: string,
  description: string,
  component: Function,
  cover: string,
|};

export type ExperimentsMap = {
  [string]: Experiment,
};
