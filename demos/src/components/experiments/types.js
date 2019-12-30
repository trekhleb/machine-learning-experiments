import type { Node } from 'react';

export type Experiment = {|
  slug: string,
  name: string,
  description: string,
  component: () => Node,
  cover: string,
  notebookUrl: ?string,
|};

export type ExperimentsMap = {
  [string]: Experiment,
};
