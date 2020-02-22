// @flow
import React from 'react';
import type { Node } from 'react';
import { useTheme } from '@material-ui/core/styles';
import { Theme } from '@material-ui/core';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from 'recharts';

export const valueKey = 'value';
export const labelKey = 'label';

export type DataRecord = {
  value: number,
  label: string,
};

type OneHotBarProps = {
  data: ?DataRecord[],
  height?: number,
};

const defaultProps = {
  height: 100,
};

const OneHotBars = (props: OneHotBarProps): Node => {
  const { data, height } = props;
  const width = '100%';

  const theme: Theme = useTheme();

  if (!data) {
    return null;
  }

  const margins = {
    left: -25, top: 1, bottom: 1, right: 1,
  };

  return (
    <ResponsiveContainer width={width} height={height}>
      <BarChart data={data} margin={margins}>
        <YAxis dataKey={valueKey} />
        <XAxis dataKey={labelKey} interval={0} />
        <Bar
          type="monotone"
          dataKey={valueKey}
          barSize={30}
          fill={theme.palette.primary.main}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};

OneHotBars.defaultProps = defaultProps;

export default OneHotBars;
