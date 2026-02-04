"""
Farrar Analytics - Deep Insights Engine
ML-driven pattern detection and $/stream variation analysis.

Operates on aggregated data (songs, months) not raw rows, so performance
stays fast even for large datasets.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Thresholds
MIN_STREAMS_FOR_RATE = 1000
MIN_MONTHS_FOR_FORECAST = 12
MIN_MONTHS_FOR_SEASONALITY = 24
MIN_ROWS_FOR_CLUSTERING = 5


class InsightsEngine:
    """Deep analytics: $/stream variation, correlations, anomalies, forecasting, segmentation."""

    def __init__(
        self,
        df: pd.DataFrame,
        streaming_df: pd.DataFrame,
        non_streaming_platforms: Set[str],
        has_upc: bool,
    ):
        self.df = df
        self.streaming_df = streaming_df
        self._non_streaming = non_streaming_platforms
        self._has_upc = has_upc

    # ------------------------------------------------------------------
    # Phase 2: $/Stream Variation Analysis
    # ------------------------------------------------------------------

    def get_per_stream_variation(self) -> Dict[str, Any]:
        """$/stream broken down by platform, country, release type, quarter, etc."""
        sdf = self.streaming_df
        if sdf.empty:
            return {"available": False, "reason": "No streaming data available"}

        total_streams = sdf['Quantity'].sum()
        total_earnings = sdf['Earnings (USD)'].sum()
        catalog_avg = round(float(total_earnings / total_streams), 4) if total_streams > 0 else 0

        result: Dict[str, Any] = {"available": True, "catalog_avg_per_stream": catalog_avg}

        # By platform (top 10 by volume, min threshold)
        plat = sdf.groupby('Store').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
        plat = plat[plat['Quantity'] >= MIN_STREAMS_FOR_RATE]
        plat['per_stream'] = (plat['Earnings (USD)'] / plat['Quantity']).round(4)
        plat = plat.sort_values('Quantity', ascending=False).head(10)
        plat.columns = ['platform', 'streams', 'earnings', 'per_stream']
        result['by_platform'] = plat.to_dict('records')

        # By country (top 15 by volume, min threshold)
        ctry = sdf.groupby('Country of Sale').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
        ctry = ctry[ctry['Quantity'] >= MIN_STREAMS_FOR_RATE]
        ctry['per_stream'] = (ctry['Earnings (USD)'] / ctry['Quantity']).round(4)
        ctry = ctry.sort_values('Quantity', ascending=False).head(15)
        ctry.columns = ['country', 'streams', 'earnings', 'per_stream']
        result['by_country'] = ctry.to_dict('records')

        # By release type
        if self._has_upc and 'release_type' in sdf.columns:
            rt = sdf.groupby('release_type').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
            rt['per_stream'] = (rt['Earnings (USD)'] / rt['Quantity']).round(4)
            rt.columns = ['release_type', 'streams', 'earnings', 'per_stream']
            rt = rt.sort_values('earnings', ascending=False)
            result['by_release_type'] = rt.to_dict('records')

        # By release name
        if self._has_upc and 'release_name' in sdf.columns:
            rn = sdf.groupby('release_name').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
            rn = rn[rn['Quantity'] >= MIN_STREAMS_FOR_RATE]
            rn['per_stream'] = (rn['Earnings (USD)'] / rn['Quantity']).round(4)
            rn.columns = ['release_name', 'streams', 'earnings', 'per_stream']
            rn = rn.sort_values('earnings', ascending=False)
            result['by_release_name'] = rn.to_dict('records')

        # By quarter
        qtr = sdf.groupby('Quarter').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
        qtr = qtr[qtr['Quantity'] >= MIN_STREAMS_FOR_RATE]
        qtr['per_stream'] = (qtr['Earnings (USD)'] / qtr['Quantity']).round(4)
        qtr.columns = ['quarter', 'streams', 'earnings', 'per_stream']
        qtr = qtr.sort_values('quarter')
        result['by_quarter'] = qtr.to_dict('records')

        # Release type x quarter
        if self._has_upc and 'release_type' in sdf.columns:
            rtq = sdf.groupby(['release_type', 'Quarter']).agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
            rtq = rtq[rtq['Quantity'] >= MIN_STREAMS_FOR_RATE]
            rtq['per_stream'] = (rtq['Earnings (USD)'] / rtq['Quantity']).round(4)
            rtq.columns = ['release_type', 'quarter', 'streams', 'earnings', 'per_stream']
            rtq = rtq.sort_values(['release_type', 'quarter'])
            result['by_release_type_quarter'] = rtq.to_dict('records')

        return self._clean(result)

    def get_song_rate_drivers(self, top_n: int = 10) -> Dict[str, Any]:
        """For top songs, explain why their $/stream is high or low."""
        sdf = self.streaming_df
        if sdf.empty:
            return {"available": False, "reason": "No streaming data available"}

        total_streams = sdf['Quantity'].sum()
        total_earnings = sdf['Earnings (USD)'].sum()
        catalog_avg = total_earnings / total_streams if total_streams > 0 else 0

        # Top songs by total earnings (full df)
        top_songs = self.df.groupby('Title')['Earnings (USD)'].sum().sort_values(ascending=False).head(top_n).index

        drivers = []
        for song in top_songs:
            song_streaming = sdf[sdf['Title'] == song]
            song_streams = song_streaming['Quantity'].sum()
            song_earnings = song_streaming['Earnings (USD)'].sum()
            song_rate = song_earnings / song_streams if song_streams > 0 else 0

            pct_diff = ((song_rate - catalog_avg) / catalog_avg * 100) if catalog_avg > 0 else 0

            # Primary platform
            plat_agg = song_streaming.groupby('Store').agg({'Quantity': 'sum'}).reset_index()
            if not plat_agg.empty:
                primary_plat = plat_agg.sort_values('Quantity', ascending=False).iloc[0]
                primary_platform = primary_plat['Store']
                primary_platform_share = round(float(primary_plat['Quantity'] / song_streams * 100), 1) if song_streams > 0 else 0
            else:
                primary_platform = None
                primary_platform_share = 0

            # Primary country
            ctry_agg = song_streaming.groupby('Country of Sale').agg({'Quantity': 'sum'}).reset_index()
            if not ctry_agg.empty:
                primary_ctry = ctry_agg.sort_values('Quantity', ascending=False).iloc[0]
                primary_country = primary_ctry['Country of Sale']
                primary_country_share = round(float(primary_ctry['Quantity'] / song_streams * 100), 1) if song_streams > 0 else 0
            else:
                primary_country = None
                primary_country_share = 0

            platform_count = int(song_streaming['Store'].nunique())

            entry: Dict[str, Any] = {
                "title": song,
                "per_stream": round(float(song_rate), 4),
                "catalog_avg": round(float(catalog_avg), 4),
                "pct_diff_from_catalog": round(float(pct_diff), 1),
                "streams": int(song_streams),
                "streaming_earnings": round(float(song_earnings), 2),
                "primary_platform": primary_platform,
                "primary_platform_share_pct": primary_platform_share,
                "primary_country": primary_country,
                "primary_country_share_pct": primary_country_share,
                "platform_count": platform_count,
            }

            # Release type info
            if self._has_upc and 'release_type' in self.df.columns:
                song_row = self.df[self.df['Title'] == song]
                if not song_row.empty:
                    rtype = song_row['release_type'].iloc[0]
                    rname = song_row['release_name'].iloc[0] if 'release_name' in song_row.columns else None
                    entry['release_type'] = rtype
                    entry['release_name'] = rname

                    # Release type avg rate
                    rt_data = sdf[sdf['release_type'] == rtype]
                    rt_streams = rt_data['Quantity'].sum()
                    rt_earnings = rt_data['Earnings (USD)'].sum()
                    rt_avg = rt_earnings / rt_streams if rt_streams > 0 else 0
                    entry['release_type_avg_per_stream'] = round(float(rt_avg), 4)

            drivers.append(entry)

        return {"available": True, "catalog_avg_per_stream": round(float(catalog_avg), 4), "songs": drivers}

    def get_platform_country_matrix(self, top_platforms: int = 5, top_countries: int = 10) -> Dict[str, Any]:
        """Best/worst platform+country combos by $/stream."""
        sdf = self.streaming_df
        if sdf.empty:
            return {"available": False, "reason": "No streaming data available"}

        # Top platforms and countries by volume
        top_plats = sdf.groupby('Store')['Quantity'].sum().sort_values(ascending=False).head(top_platforms).index
        top_ctries = sdf.groupby('Country of Sale')['Quantity'].sum().sort_values(ascending=False).head(top_countries).index

        filtered = sdf[sdf['Store'].isin(top_plats) & sdf['Country of Sale'].isin(top_ctries)]
        matrix = filtered.groupby(['Store', 'Country of Sale']).agg({
            'Quantity': 'sum', 'Earnings (USD)': 'sum'
        }).reset_index()
        matrix = matrix[matrix['Quantity'] >= MIN_STREAMS_FOR_RATE]

        if matrix.empty:
            return {"available": False, "reason": "Insufficient data for platform-country matrix"}

        matrix['per_stream'] = (matrix['Earnings (USD)'] / matrix['Quantity']).round(4)
        matrix.columns = ['platform', 'country', 'streams', 'earnings', 'per_stream']
        matrix['earnings'] = matrix['earnings'].round(2)
        matrix = matrix.sort_values('per_stream', ascending=False)

        best = matrix.head(5).to_dict('records')
        worst = matrix.tail(5).sort_values('per_stream').to_dict('records')

        return {
            "available": True,
            "combinations": matrix.to_dict('records'),
            "best": best,
            "worst": worst,
        }

    # ------------------------------------------------------------------
    # Phase 3: ML-Driven Pattern Detection
    # ------------------------------------------------------------------

    def get_correlations(self) -> Dict[str, Any]:
        """Pearson correlations on song-level features."""
        try:
            from scipy import stats
        except ImportError:
            return {"available": False, "reason": "scipy not installed"}

        sdf = self.streaming_df

        # Build song-level features from full df
        song_features = self.df.groupby('Title').agg({
            'Earnings (USD)': 'sum',
            'Quantity': 'sum',
            'Store': 'nunique',
            'Country of Sale': 'nunique',
            'Sale Month': 'nunique',
        }).reset_index()
        song_features.columns = ['title', 'earnings', 'total_qty', 'platform_count', 'country_count', 'month_count']

        # Streaming per_stream per song
        str_song = sdf.groupby('Title').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
        str_song.columns = ['title', 'streaming_streams', 'streaming_earnings']
        str_song['per_stream'] = str_song['streaming_earnings'] / str_song['streaming_streams']
        str_song['per_stream'] = str_song['per_stream'].replace([np.inf, -np.inf], np.nan)

        song_features = song_features.merge(str_song[['title', 'per_stream']], on='title', how='left')

        if len(song_features) < 5:
            return {"available": False, "reason": "Not enough songs for correlation analysis"}

        correlations = []

        pairs = [
            ('platform_count', 'earnings', 'Platform count vs earnings'),
            ('country_count', 'earnings', 'Country count vs earnings'),
            ('per_stream', 'earnings', 'Per-stream rate vs earnings'),
            ('month_count', 'earnings', 'Longevity (months) vs earnings'),
        ]

        for col_a, col_b, label in pairs:
            valid = song_features[[col_a, col_b]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid[col_a], valid[col_b])
            correlations.append({
                "pair": label,
                "r": round(float(r), 3),
                "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
            })

        # Release type correlations (if UPC available)
        if self._has_upc and 'release_type' in self.df.columns:
            # Merge release type onto song features
            rt_map = self.df.groupby('Title')['release_type'].first().reset_index()
            sf_rt = song_features.merge(rt_map, left_on='title', right_on='Title', how='left')

            # Encode release type numerically
            type_map = {'Single': 0, 'EP': 1, 'Album': 2}
            sf_rt['release_type_num'] = sf_rt['release_type'].map(type_map)
            valid_rt = sf_rt.dropna(subset=['release_type_num'])

            if len(valid_rt) >= 5:
                # release_type vs per_stream
                valid_ps = valid_rt.dropna(subset=['per_stream'])
                if len(valid_ps) >= 5:
                    r, p = stats.pearsonr(valid_ps['release_type_num'], valid_ps['per_stream'])
                    correlations.append({
                        "pair": "Release type vs per-stream rate",
                        "r": round(float(r), 3),
                        "p_value": round(float(p), 4),
                        "significant": bool(p < 0.05),
                    })

                # release_type vs earnings
                r, p = stats.pearsonr(valid_rt['release_type_num'], valid_rt['earnings'])
                correlations.append({
                    "pair": "Release type vs earnings",
                    "r": round(float(r), 3),
                    "p_value": round(float(p), 4),
                    "significant": bool(p < 0.05),
                })

            # Track count vs per_stream
            if 'track_count' in self.df.columns:
                tc_map = self.df.groupby('Title')['track_count'].first().reset_index()
                sf_tc = song_features.merge(tc_map, left_on='title', right_on='Title', how='left')
                valid_tc = sf_tc.dropna(subset=['track_count', 'per_stream'])
                if len(valid_tc) >= 5:
                    r, p = stats.pearsonr(valid_tc['track_count'], valid_tc['per_stream'])
                    correlations.append({
                        "pair": "Track count vs per-stream rate",
                        "r": round(float(r), 3),
                        "p_value": round(float(p), 4),
                        "significant": bool(p < 0.05),
                    })

            # ANOVA across release type groups for per_stream
            groups = []
            for rtype in ['Single', 'EP', 'Album']:
                grp = sf_rt.loc[sf_rt['release_type'] == rtype, 'per_stream'].dropna()
                if len(grp) >= 2:
                    groups.append(grp)
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                correlations.append({
                    "pair": "ANOVA: release type groups vs per-stream",
                    "f_statistic": round(float(f_stat), 3) if not np.isnan(f_stat) else None,
                    "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
                    "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
                })

        return {"available": True, "correlations": correlations}

    def get_anomaly_detection(self) -> Dict[str, Any]:
        """Z-score based anomaly detection on earnings and per-stream."""
        anomalies: Dict[str, Any] = {"available": True}

        # Monthly earnings anomalies (z > 2.0 on MoM change)
        monthly = self.df.groupby(self.df['Sale Month'].dt.to_period('M')).agg({
            'Earnings (USD)': 'sum'
        }).reset_index()
        monthly.columns = ['month', 'earnings']
        monthly = monthly.sort_values('month')
        monthly['mom_change'] = monthly['earnings'].diff()

        if len(monthly) >= 6:
            mean_change = monthly['mom_change'].mean()
            std_change = monthly['mom_change'].std()
            if std_change and std_change > 0:
                monthly['z_score'] = ((monthly['mom_change'] - mean_change) / std_change)
                spikes = monthly[monthly['z_score'].abs() > 2.0].copy()
                spikes['month'] = spikes['month'].astype(str)
                anomalies['monthly_earnings'] = spikes[['month', 'earnings', 'mom_change', 'z_score']].round(2).to_dict('records')
            else:
                anomalies['monthly_earnings'] = []
        else:
            anomalies['monthly_earnings'] = []

        # Per-song monthly spikes (>200% jump or >60% drop MoM)
        song_monthly = self.df.groupby([self.df['Sale Month'].dt.to_period('M'), 'Title']).agg({
            'Earnings (USD)': 'sum'
        }).reset_index()
        song_monthly.columns = ['month', 'title', 'earnings']
        song_monthly = song_monthly.sort_values(['title', 'month'])
        song_monthly['prev'] = song_monthly.groupby('title')['earnings'].shift(1)
        song_monthly['change_pct'] = ((song_monthly['earnings'] / song_monthly['prev']) - 1) * 100

        song_spikes = song_monthly[
            (song_monthly['change_pct'] > 200) | (song_monthly['change_pct'] < -60)
        ].copy()
        song_spikes = song_spikes.dropna(subset=['change_pct'])
        song_spikes['month'] = song_spikes['month'].astype(str)
        # Limit to top 20 most extreme
        song_spikes = song_spikes.reindex(song_spikes['change_pct'].abs().sort_values(ascending=False).index).head(20)
        anomalies['song_spikes'] = song_spikes[['month', 'title', 'earnings', 'change_pct']].round(2).to_dict('records')

        # Geographic spikes (country jumping >3 std devs above its mean for a song)
        geo = self.df.groupby(['Title', 'Country of Sale']).agg({
            'Earnings (USD)': 'sum'
        }).reset_index()
        geo.columns = ['title', 'country', 'earnings']

        # Per-song: mean/std across countries
        song_stats = geo.groupby('title')['earnings'].agg(['mean', 'std']).reset_index()
        song_stats.columns = ['title', 'country_mean', 'country_std']
        geo = geo.merge(song_stats, on='title')
        geo['z'] = np.where(geo['country_std'] > 0, (geo['earnings'] - geo['country_mean']) / geo['country_std'], 0)
        geo_spikes = geo[geo['z'] > 3.0].sort_values('z', ascending=False).head(15)
        anomalies['geographic_spikes'] = geo_spikes[['title', 'country', 'earnings', 'z']].round(2).to_dict('records')

        # Release-type anomalies: $/stream deviation from own historical trend
        if self._has_upc and 'release_type' in self.streaming_df.columns:
            sdf = self.streaming_df
            rtq = sdf.groupby(['release_type', sdf['Sale Month'].dt.to_period('Q')]).agg({
                'Quantity': 'sum', 'Earnings (USD)': 'sum'
            }).reset_index()
            rtq.columns = ['release_type', 'quarter', 'streams', 'earnings']
            rtq['per_stream'] = rtq['earnings'] / rtq['streams']
            rtq = rtq.sort_values(['release_type', 'quarter'])
            rtq['prev_rate'] = rtq.groupby('release_type')['per_stream'].shift(1)
            rtq['rate_change_pct'] = ((rtq['per_stream'] / rtq['prev_rate']) - 1) * 100
            rt_anomalies = rtq[rtq['rate_change_pct'].abs() > 15].dropna(subset=['rate_change_pct']).copy()
            rt_anomalies['quarter'] = rt_anomalies['quarter'].astype(str)
            anomalies['release_type_rate_shifts'] = rt_anomalies[
                ['release_type', 'quarter', 'per_stream', 'rate_change_pct']
            ].round(4).to_dict('records')
        else:
            anomalies['release_type_rate_shifts'] = []

        return self._clean(anomalies)

    def get_trend_forecast(self, periods: int = 6) -> Dict[str, Any]:
        """Time series forecast using exponential smoothing."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.api import SimpleExpSmoothing
        except ImportError:
            return {"available": False, "reason": "statsmodels not installed"}

        monthly = self.df.groupby(self.df['Sale Month'].dt.to_period('M')).agg({
            'Earnings (USD)': 'sum'
        }).reset_index()
        monthly.columns = ['month', 'earnings']
        monthly = monthly.sort_values('month')

        n_months = len(monthly)
        if n_months < MIN_MONTHS_FOR_FORECAST:
            return {"available": False, "reason": f"Need at least {MIN_MONTHS_FOR_FORECAST} months of data (have {n_months})"}

        series = monthly['earnings'].values.astype(float)
        # Ensure no zeros (smoothing can fail)
        series = np.maximum(series, 0.01)

        result: Dict[str, Any] = {"available": True}

        try:
            if n_months >= MIN_MONTHS_FOR_SEASONALITY:
                model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
                fit = model.fit(optimized=True)
            else:
                model = SimpleExpSmoothing(series)
                fit = model.fit(optimized=True)

            forecast = fit.forecast(periods)
            last_month = monthly['month'].iloc[-1]

            forecast_months = []
            for i in range(1, periods + 1):
                fm = last_month + i
                forecast_months.append({
                    "month": str(fm),
                    "predicted_earnings": round(float(forecast[i - 1]), 2),
                })
            result['forecast'] = forecast_months

            # Trend direction
            if len(series) >= 6:
                recent_avg = float(np.mean(series[-6:]))
                prior_avg = float(np.mean(series[-12:-6])) if len(series) >= 12 else float(np.mean(series[:len(series) // 2]))
                if recent_avg > prior_avg * 1.05:
                    result['trend_direction'] = 'growing'
                elif recent_avg < prior_avg * 0.95:
                    result['trend_direction'] = 'declining'
                else:
                    result['trend_direction'] = 'stable'

            # Seasonal pattern
            if n_months >= MIN_MONTHS_FOR_SEASONALITY:
                monthly_avgs = {}
                for idx, row in monthly.iterrows():
                    m = row['month'].month
                    monthly_avgs.setdefault(m, []).append(row['earnings'])
                seasonal = {m: round(float(np.mean(vals)), 2) for m, vals in monthly_avgs.items()}
                peak_month = max(seasonal, key=seasonal.get)
                low_month = min(seasonal, key=seasonal.get)
                result['seasonal_pattern'] = {
                    "monthly_averages": seasonal,
                    "peak_month": int(peak_month),
                    "low_month": int(low_month),
                }

        except Exception as e:
            logger.warning(f"Forecast model failed: {e}")
            result['forecast'] = []
            result['error'] = str(e)

        # Per-release-type forecast
        if self._has_upc and 'release_type' in self.df.columns:
            rt_forecasts = {}
            for rtype in ['Single', 'EP', 'Album']:
                rt_df = self.df[self.df['release_type'] == rtype]
                rt_monthly = rt_df.groupby(rt_df['Sale Month'].dt.to_period('M')).agg({
                    'Earnings (USD)': 'sum'
                }).reset_index()
                rt_monthly.columns = ['month', 'earnings']
                rt_monthly = rt_monthly.sort_values('month')

                if len(rt_monthly) < MIN_MONTHS_FOR_FORECAST:
                    rt_forecasts[rtype] = {"available": False, "reason": "Insufficient data"}
                    continue

                rt_series = rt_monthly['earnings'].values.astype(float)
                rt_series = np.maximum(rt_series, 0.01)
                try:
                    rt_model = SimpleExpSmoothing(rt_series)
                    rt_fit = rt_model.fit(optimized=True)
                    rt_forecast = rt_fit.forecast(periods)
                    rt_last = rt_monthly['month'].iloc[-1]
                    rt_forecasts[rtype] = {
                        "available": True,
                        "forecast": [
                            {"month": str(rt_last + i), "predicted_earnings": round(float(rt_forecast[i - 1]), 2)}
                            for i in range(1, periods + 1)
                        ],
                    }
                    # Trend
                    if len(rt_series) >= 6:
                        recent = float(np.mean(rt_series[-6:]))
                        prior = float(np.mean(rt_series[-12:-6])) if len(rt_series) >= 12 else float(np.mean(rt_series[:len(rt_series) // 2]))
                        if recent > prior * 1.05:
                            rt_forecasts[rtype]['trend'] = 'growing'
                        elif recent < prior * 0.95:
                            rt_forecasts[rtype]['trend'] = 'declining'
                        else:
                            rt_forecasts[rtype]['trend'] = 'stable'
                except Exception as e:
                    rt_forecasts[rtype] = {"available": False, "reason": str(e)}

            result['by_release_type'] = rt_forecasts

        return self._clean(result)

    def get_song_segmentation(self) -> Dict[str, Any]:
        """KMeans clustering of songs into 4 segments."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"available": False, "reason": "scikit-learn not installed"}

        sdf = self.streaming_df

        # Build song-level features
        song_full = self.df.groupby('Title').agg({
            'Earnings (USD)': 'sum',
            'Quantity': 'sum',
            'Store': 'nunique',
            'Country of Sale': 'nunique',
        }).reset_index()
        song_full.columns = ['title', 'total_earnings', 'total_qty', 'platform_count', 'country_count']

        # Streaming metrics
        str_song = sdf.groupby('Title').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum'}).reset_index()
        str_song.columns = ['title', 'total_streams', 'streaming_earnings']
        str_song['per_stream'] = (str_song['streaming_earnings'] / str_song['total_streams']).replace([np.inf, -np.inf], 0).fillna(0)

        song_full = song_full.merge(str_song[['title', 'total_streams', 'per_stream']], on='title', how='left')
        song_full['total_streams'] = song_full['total_streams'].fillna(0)
        song_full['per_stream'] = song_full['per_stream'].fillna(0)

        # Recent momentum: last 3 months earnings / total earnings
        max_date = self.df['Sale Month'].max()
        recent_start = max_date - pd.DateOffset(months=3)
        recent = self.df[self.df['Sale Month'] > recent_start].groupby('Title')['Earnings (USD)'].sum().reset_index()
        recent.columns = ['title', 'recent_earnings']
        song_full = song_full.merge(recent, on='title', how='left')
        song_full['recent_earnings'] = song_full['recent_earnings'].fillna(0)
        song_full['recent_momentum'] = np.where(
            song_full['total_earnings'] > 0,
            song_full['recent_earnings'] / song_full['total_earnings'],
            0
        )

        if len(song_full) < MIN_ROWS_FOR_CLUSTERING:
            return {"available": False, "reason": f"Need at least {MIN_ROWS_FOR_CLUSTERING} songs (have {len(song_full)})"}

        # Feature matrix
        feature_cols = ['total_earnings', 'total_streams', 'per_stream', 'platform_count', 'country_count', 'recent_momentum']

        # Add release type if available
        if self._has_upc and 'release_type' in self.df.columns:
            rt_map = self.df.groupby('Title')['release_type'].first().reset_index()
            song_full = song_full.merge(rt_map, left_on='title', right_on='Title', how='left')
            if 'Title' in song_full.columns and 'title' in song_full.columns:
                song_full = song_full.drop(columns=['Title'])
            type_encoding = {'Single': 0, 'EP': 1, 'Album': 2}
            song_full['release_type_num'] = song_full['release_type'].map(type_encoding).fillna(0)
            feature_cols.append('release_type_num')

        # Add release name if available
        if self._has_upc and 'release_name' in self.df.columns:
            rn_map = self.df.groupby('Title')['release_name'].first().reset_index()
            song_full = song_full.merge(rn_map, left_on='title', right_on='Title', how='left', suffixes=('', '_rn'))
            if 'Title' in song_full.columns and 'title' in song_full.columns:
                song_full = song_full.drop(columns=[c for c in song_full.columns if c == 'Title'], errors='ignore')

        X = song_full[feature_cols].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(4, len(song_full))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        song_full['cluster'] = kmeans.fit_predict(X_scaled)

        # Label clusters by centroid characteristics
        centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
        labels = self._label_clusters(centroids)

        song_full['segment'] = song_full['cluster'].map(labels)

        # Build segment summaries
        segments = []
        output_cols = ['title', 'total_earnings', 'total_streams', 'per_stream', 'platform_count', 'country_count', 'recent_momentum']
        if 'release_type' in song_full.columns:
            output_cols.append('release_type')
        if 'release_name' in song_full.columns:
            output_cols.append('release_name')

        for segment_name in sorted(labels.values()):
            seg_songs = song_full[song_full['segment'] == segment_name]
            seg_data = {
                "segment": segment_name,
                "count": int(len(seg_songs)),
                "total_earnings": round(float(seg_songs['total_earnings'].sum()), 2),
                "avg_per_stream": round(float(seg_songs['per_stream'].mean()), 4),
                "avg_momentum": round(float(seg_songs['recent_momentum'].mean()), 3),
                "songs": seg_songs[output_cols].round(4).to_dict('records'),
            }
            # Release type distribution
            if 'release_type' in seg_songs.columns:
                rt_dist = seg_songs['release_type'].value_counts().to_dict()
                seg_data['release_type_distribution'] = rt_dist

            segments.append(seg_data)

        return {"available": True, "segments": segments}

    def get_release_strategy_insights(self) -> Dict[str, Any]:
        """Release pattern analysis: effectiveness by type, cannibalization, variant performance."""
        if not self._has_upc:
            return {"available": False, "reason": "Requires UPC data"}

        sdf = self.streaming_df
        result: Dict[str, Any] = {"available": True}

        # Release type effectiveness
        rt_eff = []
        for rtype in ['Single', 'EP', 'Album']:
            rt_data = self.df[self.df['release_type'] == rtype]
            if rt_data.empty:
                continue
            rt_streaming = sdf[sdf['release_type'] == rtype] if 'release_type' in sdf.columns else pd.DataFrame()

            n_releases = rt_data['UPC'].nunique()
            total_earnings = rt_data['Earnings (USD)'].sum()
            avg_earnings = total_earnings / n_releases if n_releases > 0 else 0

            str_streams = rt_streaming['Quantity'].sum() if not rt_streaming.empty else 0
            str_earnings = rt_streaming['Earnings (USD)'].sum() if not rt_streaming.empty else 0
            avg_per_stream = str_earnings / str_streams if str_streams > 0 else 0

            # Avg longevity: unique months per release
            longevity = rt_data.groupby('UPC')['Sale Month'].nunique().mean()

            rt_eff.append({
                "release_type": rtype,
                "release_count": int(n_releases),
                "total_earnings": round(float(total_earnings), 2),
                "avg_earnings_per_release": round(float(avg_earnings), 2),
                "avg_per_stream": round(float(avg_per_stream), 4),
                "avg_longevity_months": round(float(longevity), 1),
            })

        result['type_effectiveness'] = rt_eff

        # Cross-released songs: cannibalization analysis
        title_releases = self.df.groupby('Title')['UPC'].nunique()
        cross_titles = title_releases[title_releases > 1].index

        cannibalization = []
        for title in cross_titles:
            title_df = self.df[self.df['Title'] == title]
            by_release = title_df.groupby(['UPC', 'release_type']).agg({
                'Quantity': 'sum', 'Earnings (USD)': 'sum'
            }).reset_index()

            total_streams = by_release['Quantity'].sum()
            if total_streams == 0:
                continue

            splits = []
            for _, row in by_release.iterrows():
                splits.append({
                    "release_type": row['release_type'],
                    "streams": int(row['Quantity']),
                    "earnings": round(float(row['Earnings (USD)']), 2),
                    "stream_share_pct": round(float(row['Quantity'] / total_streams * 100), 1),
                })

            cannibalization.append({
                "title": title,
                "total_streams": int(total_streams),
                "total_earnings": round(float(by_release['Earnings (USD)'].sum()), 2),
                "splits": splits,
            })

        cannibalization.sort(key=lambda x: x['total_earnings'], reverse=True)
        result['cannibalization'] = cannibalization[:20]

        # Variant performance: remasters, remixes, acoustic, slowed
        variant_keywords = ['remix', 'remaster', 'acoustic', 'slowed', 'sped up', 'live', 'instrumental']
        variant_rows = self.df[self.df['Title'].str.lower().str.contains('|'.join(variant_keywords), na=False)]

        if not variant_rows.empty:
            variants = []
            for title in variant_rows['Title'].unique():
                # Try to find the original
                base = title.split('(')[0].strip().split(' - ')[0].strip()
                originals = self.df[
                    (self.df['Title'].str.lower() == base.lower()) |
                    (self.df['Title'].str.lower().str.startswith(base.lower()) & ~self.df['Title'].str.lower().str.contains('|'.join(variant_keywords), na=False))
                ]
                original_title = originals['Title'].iloc[0] if not originals.empty else None

                variant_data = self.df[self.df['Title'] == title]
                v_earnings = variant_data['Earnings (USD)'].sum()
                v_streams = variant_data['Quantity'].sum()

                entry: Dict[str, Any] = {
                    "variant_title": title,
                    "earnings": round(float(v_earnings), 2),
                    "streams": int(v_streams),
                }

                if original_title:
                    orig_data = self.df[self.df['Title'] == original_title]
                    o_earnings = orig_data['Earnings (USD)'].sum()
                    entry['original_title'] = original_title
                    entry['original_earnings'] = round(float(o_earnings), 2)
                    entry['variant_pct_of_original'] = round(float(v_earnings / o_earnings * 100), 1) if o_earnings > 0 else 0

                variants.append(entry)

            variants.sort(key=lambda x: x['earnings'], reverse=True)
            result['variant_performance'] = variants[:15]
        else:
            result['variant_performance'] = []

        # Release type trend: is the artist shifting release strategy over time?
        if 'release_type' in self.df.columns:
            yearly_rt = self.df.groupby([self.df['Sale Month'].dt.year, 'release_type']).agg({
                'Earnings (USD)': 'sum',
                'UPC': 'nunique',
            }).reset_index()
            yearly_rt.columns = ['year', 'release_type', 'earnings', 'release_count']
            yearly_rt['earnings'] = yearly_rt['earnings'].round(2)
            yearly_rt = yearly_rt.sort_values(['year', 'release_type'])
            result['release_type_trend'] = yearly_rt.to_dict('records')

        return self._clean(result)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def get_all_insights(self) -> Dict[str, Any]:
        """Run all insight methods with per-section error handling."""
        sections = {
            "per_stream_variation": self.get_per_stream_variation,
            "song_rate_drivers": self.get_song_rate_drivers,
            "platform_country_matrix": self.get_platform_country_matrix,
            "correlations": self.get_correlations,
            "anomaly_detection": self.get_anomaly_detection,
            "trend_forecast": self.get_trend_forecast,
            "song_segmentation": self.get_song_segmentation,
            "release_strategy": self.get_release_strategy_insights,
        }

        results: Dict[str, Any] = {}
        for key, method in sections.items():
            try:
                results[key] = method()
            except Exception as e:
                logger.error(f"InsightsEngine.{key} failed: {e}", exc_info=True)
                results[key] = {"available": False, "error": str(e)}

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_clusters(centroids: pd.DataFrame) -> Dict[int, str]:
        """Assign human-readable labels to clusters based on centroid characteristics."""
        labels = {}
        n = len(centroids)
        available_labels = ["Cash Cows", "Rising Stars", "Niche Performers", "Underperformers"]

        # Rank clusters by total_earnings
        earnings_rank = centroids['total_earnings'].rank(ascending=False).astype(int)
        # Rank by recent_momentum
        momentum_rank = centroids['recent_momentum'].rank(ascending=False).astype(int)
        # Rank by per_stream
        rate_rank = centroids['per_stream'].rank(ascending=False).astype(int)

        for i in range(n):
            if n == 1:
                labels[i] = "Catalog"
            elif earnings_rank[i] == 1:
                labels[i] = "Cash Cows"
            elif momentum_rank[i] == 1 and earnings_rank[i] != 1:
                labels[i] = "Rising Stars"
            elif rate_rank[i] <= 2 and earnings_rank[i] > 2:
                labels[i] = "Niche Performers"
            else:
                labels[i] = "Underperformers"

        # Deduplicate labels
        used = set()
        for i in sorted(labels.keys()):
            if labels[i] in used:
                # Assign next available
                for lbl in available_labels:
                    if lbl not in used:
                        labels[i] = lbl
                        break
                else:
                    labels[i] = f"Segment {i + 1}"
            used.add(labels[i])

        return labels

    @staticmethod
    def _clean(obj: Any) -> Any:
        """Recursively clean NaN/Inf/numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: InsightsEngine._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [InsightsEngine._clean(item) for item in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return obj
