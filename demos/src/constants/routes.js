// @flow
import { ML_EXPERIMENTS_GITHUB_REPO_NAME } from './links';

export const ROOT_ROUTE = '/';
export const HOME_ROUTE = `/${ML_EXPERIMENTS_GITHUB_REPO_NAME}`;

export const EXPERIMENT_ID_PARAM = 'experiment_id';
export const EXPERIMENTS_ROUTE = `/${ML_EXPERIMENTS_GITHUB_REPO_NAME}/experiments`;
export const EXPERIMENT_ROUTE = `${EXPERIMENTS_ROUTE}/:${EXPERIMENT_ID_PARAM}`;
