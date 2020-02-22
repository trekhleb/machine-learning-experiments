// @flow
import type { Location } from 'history';
import { GOOGLE_ANALYTICS_ID } from '../constants/analytics';

// eslint-disable-next-line
export const googleAnalyticsTrack = (location: Location): void => {
  if (window.gtag) {
    window.gtag('config', GOOGLE_ANALYTICS_ID, {
      page_path: `${location.pathname}${location.search}`,
    });
  }
};
