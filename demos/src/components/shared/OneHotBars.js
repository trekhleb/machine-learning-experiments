import React from 'react';
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

const OneHotBars = (props: OneHotBarProps) => {
  const {data} = props;

  if (!data) {
    return null;
  }
  
  return (
    <ResponsiveContainer
      width="100%"
      height={100}
      margin={{top: 0, right: 0, bottom: 0, left: 0}}
    >
      <BarChart data={data}>
        <XAxis dataKey={labelKey} />
        <YAxis />
        <Bar type="monotone" dataKey={valueKey} barSize={30} fill="#8884d8" />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default OneHotBars;
