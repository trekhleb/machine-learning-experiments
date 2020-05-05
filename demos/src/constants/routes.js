// @flow
export const ROOT_ROUTE = (process && process.env && process.env.PUBLIC_URL) || '';
export const HOME_ROUTE = `${ROOT_ROUTE}`;

export const EXPERIMENT_ID_PARAM = 'experiment_id';
export const EXPERIMENTS_ROUTE = `${ROOT_ROUTE}/experiments`;
export const EXPERIMENT_ROUTE = `${EXPERIMENTS_ROUTE}/:${EXPERIMENT_ID_PARAM}`;
