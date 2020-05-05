// @flow
import React from 'react';

const jupyterBadgeUrl = 'https://mybinder.org/badge_logo.svg';
const colabBadgeUrl = 'https://colab.research.google.com/assets/colab-badge.svg';

// @see: https://shields.io/category/build
// const githubBadgeUrl = 'https://img.shields.io/static/v1?label=GitHub&message=Fork&color=green&logo=github&style=flat';
// const githubBadgeUrl = 'https://img.shields.io/github/stars/trekhleb/machine-learning-experiments?label=Star&style=social';
const githubBadgeUrl = 'https://img.shields.io/github/stars/trekhleb/machine-learning-experiments?style=social';

export const badgeType = {
  colab: 'colab',
  jupyter: 'jupyter',
  github: 'github',
};

type BadgeProps = {
  url: string,
  type: $Values<typeof badgeType>,
};

const Badge = (props: BadgeProps) => {
  const { url, type } = props;

  let badge = null;

  if (type === badgeType.colab) {
    badge = <img src={colabBadgeUrl} alt="Open in Colab" />;
  }

  if (type === badgeType.jupyter) {
    badge = <img src={jupyterBadgeUrl} alt="Launch in Binder" />;
  }

  if (type === badgeType.github) {
    badge = <img src={githubBadgeUrl} alt="Open on Binder" />;
  }

  return (
    <a href={url}>
      {badge}
    </a>
  );
};

export default Badge;
