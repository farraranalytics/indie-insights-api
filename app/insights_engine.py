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
MIN_STREAMS_FOR_RATE_ANALYSIS = 10000
MIN_EARNINGS_FOR_MATRIX = 50
MIN_PER_STREAM_FOR_MATRIX = 0.0001
MIN_EARNINGS_FOR_ANOMALY = 500
MIN_EARNINGS_FOR_SONG_ANOMALY = 50
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
        total_gross = sdf['gross_earnings'].sum()
        catalog_avg = round(float(total_gross / total_streams), 4) if total_streams > 0 else 0

        result: Dict[str, Any] = {"available": True, "catalog_avg_per_stream": catalog_avg}

        # By platform (top 10 by volume, min threshold)
        plat = sdf.groupby('Store').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'}).reset_index()
        plat = plat[plat['Quantity'] >= MIN_STREAMS_FOR_RATE]
        plat['per_stream'] = (plat['gross_earnings'] / plat['Quantity']).round(4)
        plat = plat.sort_values('Quantity', ascending=False).head(10)
        plat = plat.drop(columns=['gross_earnings'])
        plat.columns = ['platform', 'streams', 'earnings', 'per_stream']
        result['by_platform'] = plat.to_dict('records')

        # By country (top 15 by volume, min threshold)
        ctry = sdf.groupby('Country of Sale').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'}).reset_index()
        ctry = ctry[ctry['Quantity'] >= MIN_STREAMS_FOR_RATE]
        ctry['per_stream'] = (ctry['gross_earnings'] / ctry['Quantity']).round(4)
        ctry = ctry.sort_values('Quantity', ascending=False).head(15)
        ctry = ctry.drop(columns=['gross_earnings'])
        ctry.columns = ['country', 'streams', 'earnings', 'per_stream']
        result['by_country'] = ctry.to_dict('records')

        # By release type
        if self._has_upc and 'release_type' in sdf.columns:
            rt = sdf.groupby('release_type').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'}).reset_index()
            rt['per_stream'] = (rt['gross_earnings'] / rt['Quantity']).round(4)
            rt = rt.drop(columns=['gross_earnings'])
            rt.columns = ['release_type', 'streams', 'earnings', 'per_stream']
            rt = rt.sort_values('earnings', ascending=False)
            result['by_release_type'] = rt.to_dict('records')

        # By release name
        if self._has_upc and 'release_name' in sdf.columns:
            rn = sdf.groupby('release_name').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'}).reset_index()
            rn = rn[rn['Quantity'] >= MIN_STREAMS_FOR_RATE]
            rn['per_stream'] = (rn['gross_earnings'] / rn['Quantity']).round(4)
            rn = rn.drop(columns=['gross_earnings'])
            rn.columns = ['release_name', 'streams', 'earnings', 'per_stream']
            rn = rn.sort_values('earnings', ascending=False)
            result['by_release_name'] = rn.to_dict('records')

        # By quarter
        qtr = sdf.groupby('Quarter').agg({'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'}).reset_index()
        qtr = qtr[qtr['Quantity'] >= MIN_STREAMS_FOR_RATE]
        qtr['per_stream'] = (qtr['gross_earnings'] / qtr['Quantity']).round(4)
        qtr = qtr.drop(columns=['gross_earnings'])
        qtr.columns = ['quarter', 'streams', 'earnings', 'per_stream']
        qtr = qtr.sort_values('quarter')
        result['by_quarter'] = qtr.to_dict('records')

        # Release type x quarter
        if self._has_upc and 'release_type' in sdf.columns:
            rtq = sdf.groupby(['release_type', 'Quarter']).agg({'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'}).reset_index()
            rtq = rtq[rtq['Quantity'] >= MIN_STREAMS_FOR_RATE]
            rtq['per_stream'] = (rtq['gross_earnings'] / rtq['Quantity']).round(4)
            rtq = rtq.drop(columns=['gross_earnings'])
            rtq.columns = ['release_type', 'quarter', 'streams', 'earnings', 'per_stream']
            rtq = rtq.sort_values(['release_type', 'quarter'])
            result['by_release_type_quarter'] = rtq.to_dict('records')

        # By platform x month (for time-filtered $/stream charts)
        plat_monthly = sdf.groupby(['Store', sdf['Sale Month'].dt.strftime('%Y-%m')]).agg({
            'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'
        }).reset_index()
        plat_monthly.columns = ['platform', 'month', 'streams', 'earnings', 'gross_earnings']
        plat_monthly = plat_monthly[plat_monthly['streams'] >= MIN_STREAMS_FOR_RATE]
        plat_monthly['per_stream'] = (plat_monthly['gross_earnings'] / plat_monthly['streams']).round(4)
        plat_monthly = plat_monthly.drop(columns=['gross_earnings'])
        plat_monthly['earnings'] = plat_monthly['earnings'].round(2)
        plat_monthly = plat_monthly.sort_values(['platform', 'month'])
        result['by_platform_monthly'] = plat_monthly.to_dict('records')

        # By country x month (for time-filtered $/stream charts)
        ctry_monthly = sdf.groupby(['Country of Sale', sdf['Sale Month'].dt.strftime('%Y-%m')]).agg({
            'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'
        }).reset_index()
        ctry_monthly.columns = ['country', 'month', 'streams', 'earnings', 'gross_earnings']
        ctry_monthly = ctry_monthly[ctry_monthly['streams'] >= MIN_STREAMS_FOR_RATE]
        ctry_monthly['per_stream'] = (ctry_monthly['gross_earnings'] / ctry_monthly['streams']).round(4)
        ctry_monthly = ctry_monthly.drop(columns=['gross_earnings'])
        ctry_monthly['earnings'] = ctry_monthly['earnings'].round(2)
        ctry_monthly = ctry_monthly.sort_values(['country', 'month'])
        result['by_country_monthly'] = ctry_monthly.to_dict('records')

        return self._clean(result)

    def get_song_rate_drivers(self, top_n: int = 10) -> Dict[str, Any]:
        """For top songs, explain why their $/stream is high or low."""
        sdf = self.streaming_df
        if sdf.empty:
            return {"available": False, "reason": "No streaming data available"}

        total_streams = sdf['Quantity'].sum()
        total_gross = sdf['gross_earnings'].sum()
        catalog_avg = total_gross / total_streams if total_streams > 0 else 0

        # Filter to songs with meaningful streaming volume, then take top by earnings
        song_stream_totals = sdf.groupby('Title')['Quantity'].sum()
        eligible_songs = song_stream_totals[song_stream_totals >= MIN_STREAMS_FOR_RATE_ANALYSIS].index
        top_songs = self.df[self.df['Title'].isin(eligible_songs)].groupby('Title')['Earnings (USD)'].sum().sort_values(ascending=False).head(top_n).index

        drivers = []
        for song in top_songs:
            song_streaming = sdf[sdf['Title'] == song]
            song_streams = song_streaming['Quantity'].sum()
            song_gross = song_streaming['gross_earnings'].sum()
            song_rate = song_gross / song_streams if song_streams > 0 else 0

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
                "streaming_earnings": round(float(song_streaming['Earnings (USD)'].sum()), 2),
                "primary_platform": primary_platform,
                "primary_platform_share_pct": primary_platform_share,
                "primary_country": primary_country,
                "primary_country_share_pct": primary_country_share,
                "platform_count": platform_count,
            }

            # Per-platform breakdown for this song
            plat_detail = song_streaming.groupby('Store').agg({
                'Quantity': 'sum', 'gross_earnings': 'sum', 'Earnings (USD)': 'sum'
            }).reset_index()
            plat_detail = plat_detail[plat_detail['Quantity'] >= 100]
            plat_detail['per_stream'] = (plat_detail['gross_earnings'] / plat_detail['Quantity']).round(4)
            plat_detail['pct_of_streams'] = (plat_detail['Quantity'] / song_streams * 100).round(1) if song_streams > 0 else 0
            plat_detail = plat_detail.sort_values('Quantity', ascending=False)
            entry['by_platform'] = plat_detail[['Store', 'Quantity', 'Earnings (USD)', 'per_stream', 'pct_of_streams']].rename(
                columns={'Store': 'platform', 'Quantity': 'streams', 'Earnings (USD)': 'earnings'}
            ).round({'earnings': 2}).to_dict('records')

            # Per-country breakdown for this song (top 15)
            ctry_detail = song_streaming.groupby('Country of Sale').agg({
                'Quantity': 'sum', 'gross_earnings': 'sum', 'Earnings (USD)': 'sum'
            }).reset_index()
            ctry_detail = ctry_detail[ctry_detail['Quantity'] >= 100]
            ctry_detail['per_stream'] = (ctry_detail['gross_earnings'] / ctry_detail['Quantity']).round(4)
            ctry_detail['pct_of_streams'] = (ctry_detail['Quantity'] / song_streams * 100).round(1) if song_streams > 0 else 0
            ctry_detail = ctry_detail.sort_values('Quantity', ascending=False).head(15)
            entry['by_country'] = ctry_detail[['Country of Sale', 'Quantity', 'Earnings (USD)', 'per_stream', 'pct_of_streams']].rename(
                columns={'Country of Sale': 'country', 'Quantity': 'streams', 'Earnings (USD)': 'earnings'}
            ).round({'earnings': 2}).to_dict('records')

            # Release type info
            if self._has_upc and 'release_type' in self.df.columns:
                song_row = self.df[self.df['Title'] == song]
                if not song_row.empty:
                    rtype = song_row['release_type'].iloc[0]
                    rname = song_row['release_name'].iloc[0] if 'release_name' in song_row.columns else None
                    entry['release_type'] = rtype
                    entry['release_name'] = rname

                    # Release type avg rate (gross for true rate)
                    rt_data = sdf[sdf['release_type'] == rtype]
                    rt_streams = rt_data['Quantity'].sum()
                    rt_gross = rt_data['gross_earnings'].sum()
                    rt_avg = rt_gross / rt_streams if rt_streams > 0 else 0
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
            'Quantity': 'sum', 'Earnings (USD)': 'sum', 'gross_earnings': 'sum'
        }).reset_index()
        matrix = matrix[matrix['Quantity'] >= MIN_STREAMS_FOR_RATE]

        if matrix.empty:
            return {"available": False, "reason": "Insufficient data for platform-country matrix"}

        matrix['per_stream'] = (matrix['gross_earnings'] / matrix['Quantity']).round(4)
        matrix = matrix.drop(columns=['gross_earnings'])
        matrix.columns = ['platform', 'country', 'streams', 'earnings', 'per_stream']
        # Filter out noise: minimum earnings and per_stream thresholds
        matrix = matrix[(matrix['earnings'] >= MIN_EARNINGS_FOR_MATRIX) & (matrix['per_stream'] >= MIN_PER_STREAM_FOR_MATRIX)]
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

        # Streaming per_stream per song (gross for true rate)
        str_song = sdf.groupby('Title').agg({'Quantity': 'sum', 'gross_earnings': 'sum'}).reset_index()
        str_song.columns = ['title', 'streaming_streams', 'streaming_gross']
        str_song['per_stream'] = str_song['streaming_gross'] / str_song['streaming_streams']
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
            correlations.append(self._interpret_correlation(label, float(r), float(p)))

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
                    correlations.append(self._interpret_correlation("Release type vs per-stream rate", float(r), float(p)))

                # release_type vs earnings
                r, p = stats.pearsonr(valid_rt['release_type_num'], valid_rt['earnings'])
                correlations.append(self._interpret_correlation("Release type vs earnings", float(r), float(p)))

            # Track count vs per_stream
            if 'track_count' in self.df.columns:
                tc_map = self.df.groupby('Title')['track_count'].first().reset_index()
                sf_tc = song_features.merge(tc_map, left_on='title', right_on='Title', how='left')
                valid_tc = sf_tc.dropna(subset=['track_count', 'per_stream'])
                if len(valid_tc) >= 5:
                    r, p = stats.pearsonr(valid_tc['track_count'], valid_tc['per_stream'])
                    correlations.append(self._interpret_correlation("Track count vs per-stream rate", float(r), float(p)))

            # ANOVA across release type groups for per_stream
            groups = []
            for rtype in ['Single', 'EP', 'Album']:
                grp = sf_rt.loc[sf_rt['release_type'] == rtype, 'per_stream'].dropna()
                if len(grp) >= 2:
                    groups.append(grp)
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                p_safe = float(p_val) if not np.isnan(p_val) else 1.0
                correlations.append({
                    "pair": "ANOVA: release type groups vs per-stream",
                    "finding": "Per-stream rates differ significantly across release types" if p_safe < 0.05
                               else "Per-stream rates are similar across release types",
                    "confidence": "high" if p_safe < 0.01 else ("medium" if p_safe < 0.05 else "low"),
                    "actionable": bool(p_safe < 0.05),
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
                spikes = monthly[(monthly['z_score'].abs() > 2.0) & (monthly['earnings'] >= MIN_EARNINGS_FOR_ANOMALY)].copy()
                spikes['month'] = spikes['month'].astype(str)
                spikes['severity'] = spikes['z_score'].abs().apply(
                    lambda z: 'high' if z > 3.0 else ('medium' if z > 2.5 else 'low')
                )
                spikes['description'] = spikes.apply(
                    lambda row: f"{'Spike' if row['mom_change'] > 0 else 'Drop'} of ${abs(row['mom_change']):.0f} in {row['month']}"
                    f" ({'+' if row['mom_change'] > 0 else ''}{row['mom_change']:.0f} vs prior month)",
                    axis=1
                )
                anomalies['monthly_earnings'] = spikes[['month', 'earnings', 'mom_change', 'severity', 'description']].round(2).to_dict('records')
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
            ((song_monthly['change_pct'] > 200) | (song_monthly['change_pct'] < -60))
            & (song_monthly['earnings'] >= MIN_EARNINGS_FOR_SONG_ANOMALY)
        ].copy()
        song_spikes = song_spikes.dropna(subset=['change_pct'])
        song_spikes['month'] = song_spikes['month'].astype(str)
        # Limit to top 20 most extreme
        song_spikes = song_spikes.reindex(song_spikes['change_pct'].abs().sort_values(ascending=False).index).head(20)
        song_spikes['severity'] = song_spikes['change_pct'].abs().apply(
            lambda p: 'high' if p > 500 else ('medium' if p > 200 else 'low')
        )
        song_spikes['description'] = song_spikes.apply(
            lambda row: f"{row['title']}: {'surged' if row['change_pct'] > 0 else 'dropped'} "
            f"{abs(row['change_pct']):.0f}% in {row['month']}",
            axis=1
        )
        anomalies['song_spikes'] = song_spikes[['month', 'title', 'earnings', 'change_pct', 'severity', 'description']].round(2).to_dict('records')

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
        geo_spikes = geo[(geo['z'] > 3.0) & (geo['earnings'] >= MIN_EARNINGS_FOR_SONG_ANOMALY)].sort_values('z', ascending=False).head(15).copy()
        geo_spikes['severity'] = geo_spikes['z'].apply(
            lambda z: 'high' if z > 5.0 else ('medium' if z > 4.0 else 'low')
        )
        geo_spikes['description'] = geo_spikes.apply(
            lambda row: f"{row['title']} earned ${row['earnings']:.0f} in {row['country']}, "
            f"far above this song's country average",
            axis=1
        )
        anomalies['geographic_spikes'] = geo_spikes[['title', 'country', 'earnings', 'severity', 'description']].round(2).to_dict('records')

        # Release-type anomalies: $/stream deviation from own historical trend
        if self._has_upc and 'release_type' in self.streaming_df.columns:
            sdf = self.streaming_df
            rtq = sdf.groupby(['release_type', sdf['Sale Month'].dt.to_period('Q')]).agg({
                'Quantity': 'sum', 'gross_earnings': 'sum'
            }).reset_index()
            rtq.columns = ['release_type', 'quarter', 'streams', 'gross']
            rtq['per_stream'] = rtq['gross'] / rtq['streams']
            rtq = rtq.sort_values(['release_type', 'quarter'])
            rtq['prev_rate'] = rtq.groupby('release_type')['per_stream'].shift(1)
            rtq['rate_change_pct'] = ((rtq['per_stream'] / rtq['prev_rate']) - 1) * 100
            rt_anomalies = rtq[rtq['rate_change_pct'].abs() > 15].dropna(subset=['rate_change_pct']).copy()
            rt_anomalies['quarter'] = rt_anomalies['quarter'].astype(str)
            rt_anomalies['severity'] = rt_anomalies['rate_change_pct'].abs().apply(
                lambda p: 'high' if p > 30 else ('medium' if p > 20 else 'low')
            )
            rt_anomalies['description'] = rt_anomalies.apply(
                lambda row: f"{row['release_type']} $/stream {'rose' if row['rate_change_pct'] > 0 else 'fell'} "
                f"{abs(row['rate_change_pct']):.0f}% in {row['quarter']}",
                axis=1
            )
            anomalies['release_type_rate_shifts'] = rt_anomalies[
                ['release_type', 'quarter', 'per_stream', 'rate_change_pct', 'severity', 'description']
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

        # Streaming metrics (gross for true $/stream rate)
        str_song = sdf.groupby('Title').agg({'Quantity': 'sum', 'gross_earnings': 'sum'}).reset_index()
        str_song.columns = ['title', 'total_streams', 'streaming_gross']
        str_song['per_stream'] = (str_song['streaming_gross'] / str_song['total_streams']).replace([np.inf, -np.inf], 0).fillna(0)

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
        """Release pattern analysis: variant performance and release type trend.

        Note: type_effectiveness and cannibalization are served by
        analytics_engine.get_release_strategy_analysis() to avoid duplication.
        """
        if not self._has_upc:
            return {"available": False, "reason": "Requires UPC data"}

        result: Dict[str, Any] = {"available": True}

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
    def _interpret_correlation(label: str, r: float, p: float) -> Dict[str, Any]:
        """Convert a statistical correlation into plain English output."""
        strength = abs(r)
        direction = "positive" if r > 0 else "negative"

        if strength > 0.7:
            strength_word = "Strong"
        elif strength > 0.4:
            strength_word = "Moderate"
        elif strength > 0.2:
            strength_word = "Weak"
        else:
            strength_word = "Negligible"

        confidence = "high" if p < 0.01 else ("medium" if p < 0.05 else "low")
        actionable = strength > 0.3 and p < 0.05

        if strength_word == "Negligible":
            finding = f"No meaningful relationship found between {label.lower()}"
        else:
            finding = f"{strength_word} {direction} relationship between {label.lower()}"

        return {
            "pair": label,
            "finding": finding,
            "confidence": confidence,
            "actionable": actionable,
        }

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
