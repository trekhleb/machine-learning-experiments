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

type DataRecord = {
  value: number,
  label: string,
};

type OneHotBarProps = {
  data: ?DataRecord[],
};

const OneHotBars = (props: OneHotBarProps): Node => {
  const { data } = props;

  const theme: Theme = useTheme();

  if (!data) {
    return null;
  }

  return (
    <ResponsiveContainer width="100%" height={100}>
      <BarChart data={data}>
        <YAxis />
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

export default OneHotBars;
