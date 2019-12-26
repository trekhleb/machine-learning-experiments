import React from 'react';
import type { Node } from 'react';
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

const defaultProps = {
  barColor: '#8884d8',
};

type OneHotBarProps = {
  data: ?DataRecord[],
  barColor?: string,
};

const OneHotBars = (props: OneHotBarProps): Node => {
  const { data, barColor } = props;

  if (!data) {
    return null;
  }

  return (
    <ResponsiveContainer width="100%" height={100}>
      <BarChart data={data}>
        <YAxis />
        <XAxis dataKey={labelKey} />
        <Bar type="monotone" dataKey={valueKey} barSize={30} fill={barColor} />
      </BarChart>
    </ResponsiveContainer>
  );
};

OneHotBars.defaultProps = defaultProps;

export default OneHotBars;
