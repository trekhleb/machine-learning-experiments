export type ExperimentProps = {};

export type Experiment = {|
  slug: string,
  name: string,
  description: string,
  component: function,
  trainingURL: string,
  cover: string,
|};

export type ExperimentsMap = {
  [string]: Experiment,
};
