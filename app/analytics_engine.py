"""
Farrar Analytics - Analytics Engine
Core processing logic for DistroKid data

This is the heart of the product - converts raw DistroKid exports into actionable insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


class DistroKidAnalyzer:
    """
    Analyzes DistroKid 'Excruciating Detail' exports.

    Usage:
        df = pd.read_excel('distrokid_export.xlsx')
        analyzer = DistroKidAnalyzer(df)
        results = analyzer.get_full_analysis()
    """

    # Known download/purchase platforms (not streaming)
    NON_STREAMING_PLATFORMS = {'iTunes', 'Amazon (Downloads)', 'iTunes Songs', 'Amazon Songs'}

    # Threshold: platforms with avg $/unit above this are likely purchase platforms
    PURCHASE_RATE_THRESHOLD = 0.50

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
        self._build_streaming_split()
        self._has_upc = 'UPC' in self.df.columns
        if self._has_upc:
            self._build_release_classification()
            # Rebuild streaming split since _build_release_classification merges new columns onto self.df
            self._build_streaming_split()
    
    def _prepare_data(self):
        """Clean and prepare the dataframe"""
        # Convert Sale Month to datetime
        self.df['Sale Month'] = pd.to_datetime(self.df['Sale Month'])

        # Ensure numeric columns
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce').fillna(0)
        self.df['Earnings (USD)'] = pd.to_numeric(self.df['Earnings (USD)'], errors='coerce').fillna(0)

        # Add derived columns
        self.df['Year'] = self.df['Sale Month'].dt.year
        self.df['Quarter'] = self.df['Sale Month'].dt.to_period('Q').astype(str)

        # Team Percentage / ownership processing
        if 'Team Percentage' in self.df.columns:
            self.df['team_percentage'] = pd.to_numeric(self.df['Team Percentage'], errors='coerce').fillna(100.0)
            self._has_team_pct = True
        else:
            self.df['team_percentage'] = 100.0
            self._has_team_pct = False

        # Replace 0% with 100% (invalid data)
        self.df.loc[self.df['team_percentage'] <= 0, 'team_percentage'] = 100.0

        # Gross earnings = what the song actually generated before splits
        self.df['gross_earnings'] = self.df['Earnings (USD)'] / (self.df['team_percentage'] / 100.0)
    
    def _build_streaming_split(self):
        """Identify non-streaming platforms and build streaming/purchase DataFrames."""
        # Start with known non-streaming platforms
        self._non_streaming = set(self.NON_STREAMING_PLATFORMS)

        # Dynamically detect additional purchase platforms by avg $/unit
        platform_agg = self.df.groupby('Store').agg({'Earnings (USD)': 'sum', 'Quantity': 'sum'})
        platform_rates = (platform_agg['Earnings (USD)'] / platform_agg['Quantity']).replace([np.inf, -np.inf], 0).fillna(0)
        dynamic_purchases = set(platform_rates[platform_rates > self.PURCHASE_RATE_THRESHOLD].index)
        self._non_streaming |= dynamic_purchases

        # Only keep platforms that actually exist in the data
        self._non_streaming &= set(self.df['Store'].unique())

        # Build filtered DataFrames
        self._streaming_df = self.df[~self.df['Store'].isin(self._non_streaming)]
        self._purchase_df = self.df[self.df['Store'].isin(self._non_streaming)]

    # Platforms where album/EP-level purchases appear (not individual songs)
    PURCHASE_PLATFORMS = {'iTunes', 'Amazon (Downloads)', 'iTunes Songs'}

    def _build_release_classification(self):
        """Build release type lookup from UPC → track count → classification.

        Detects album/EP names by finding titles that ONLY appear on purchase
        platforms within a UPC. These are album-level purchase entries, not real
        tracks. They get used as the release_name and filtered from track counts.
        """
        # Ensure UPC is string for consistent grouping/serialization
        self.df['UPC'] = self.df['UPC'].astype(str)

        # Drop rows with missing/empty UPC
        valid_mask = self.df['UPC'].notna() & (self.df['UPC'] != '') & (self.df['UPC'] != 'nan')
        if not valid_mask.any():
            self._has_upc = False
            return

        valid_df = self.df.loc[valid_mask]

        # For each UPC, find titles that only appear on purchase platforms.
        # These are album/EP name entries, not actual songs.
        album_name_rows = {}  # UPC -> set of album-name titles
        for upc, group in valid_df.groupby('UPC'):
            album_titles = set()
            for title, tgroup in group.groupby('Title'):
                stores = set(tgroup['Store'].unique())
                if stores.issubset(self.PURCHASE_PLATFORMS):
                    album_titles.add(title)
            # Only treat as album names if the UPC has OTHER titles too
            all_titles = set(group['Title'].unique())
            if album_titles and (all_titles - album_titles):
                album_name_rows[upc] = album_titles

        # Build a mask for album-name rows (to exclude from track counting)
        is_album_name = self.df.apply(
            lambda row: row['Title'] in album_name_rows.get(str(row['UPC']), set()),
            axis=1
        )
        self.df['is_album_name_entry'] = is_album_name

        # Count unique REAL titles per UPC (excluding album-name entries)
        real_tracks = self.df.loc[valid_mask & ~is_album_name]
        upc_tracks = real_tracks.groupby('UPC')['Title'].nunique().reset_index()
        upc_tracks.columns = ['UPC', 'track_count']

        # Classify: Single (1), EP (2-6), Album (7+)
        def classify(count: int) -> str:
            if count <= 1:
                return 'Single'
            elif count <= 6:
                return 'EP'
            else:
                return 'Album'

        upc_tracks['release_type'] = upc_tracks['track_count'].apply(classify)

        # Build release name:
        # - If we found a purchase-only album name, use it
        # - Otherwise fall back to the first song title (for singles)
        def get_release_name(upc):
            names = album_name_rows.get(upc)
            if names:
                return sorted(names)[0]
            # Fall back to first real title alphabetically
            titles = real_tracks.loc[real_tracks['UPC'] == upc, 'Title'].unique()
            return sorted(titles)[0] if len(titles) > 0 else upc

        upc_tracks['release_name'] = upc_tracks['UPC'].apply(get_release_name)

        self._release_lookup = upc_tracks.set_index('UPC')

        # Merge release_type, track_count, and release_name onto the main dataframe
        self.df = self.df.merge(
            upc_tracks[['UPC', 'release_type', 'track_count', 'release_name']],
            on='UPC',
            how='left'
        )
        # Rows with invalid UPC get 'Unknown' release_type
        self.df['release_type'] = self.df['release_type'].fillna('Unknown')

    def get_release_breakdown(self) -> Dict[str, Any]:
        """Get release metadata: each UPC with its classification, earnings, and streams."""
        if not self._has_upc:
            return {"available": False, "reason": "No UPC column found in data"}

        releases = self.df.groupby('UPC').agg({
            'Title': 'nunique',
            'Quantity': 'sum',
            'Earnings (USD)': 'sum',
            'release_type': 'first',
            'track_count': 'first',
        }).reset_index()

        # Add release name
        release_names = self._release_lookup['release_name']
        releases = releases.merge(
            release_names.reset_index(),
            on='UPC',
            how='left'
        )

        releases = releases.rename(columns={
            'Title': 'unique_tracks',
            'Quantity': 'streams',
            'Earnings (USD)': 'earnings',
        })
        releases['earnings'] = releases['earnings'].round(2)
        releases['streams'] = releases['streams'].astype(int)
        releases['track_count'] = releases['track_count'].astype(int)
        releases = releases.sort_values('earnings', ascending=False)

        # Detect cross-released songs (titles appearing on multiple UPCs)
        title_upc = self.df.groupby('Title')['UPC'].nunique()
        cross_released = title_upc[title_upc > 1].reset_index()
        cross_released.columns = ['title', 'release_count']
        cross_released = cross_released.sort_values('release_count', ascending=False)

        return {
            "available": True,
            "releases": releases[['UPC', 'release_name', 'release_type', 'track_count',
                                   'streams', 'earnings']].to_dict('records'),
            "cross_released_songs": cross_released.to_dict('records'),
        }

    def get_release_type_summary(self) -> Dict[str, Any]:
        """Summarize performance by release type (Single / EP / Album)."""
        if not self._has_upc:
            return {"available": False, "reason": "No UPC column found in data"}

        total_earnings = self.df['Earnings (USD)'].sum()
        total_streams = self.df['Quantity'].sum()

        summary = self.df.groupby('release_type').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum',
            'UPC': 'nunique',
            'Title': 'nunique',
        }).reset_index()

        summary.columns = ['release_type', 'streams', 'earnings', 'release_count', 'track_count']
        summary['earnings'] = summary['earnings'].round(2)
        summary['streams'] = summary['streams'].astype(int)
        summary['release_count'] = summary['release_count'].astype(int)
        summary['track_count'] = summary['track_count'].astype(int)
        summary['pct_of_earnings'] = (summary['earnings'] / total_earnings * 100).round(1) if total_earnings > 0 else 0
        summary['pct_of_streams'] = (summary['streams'] / total_streams * 100).round(1) if total_streams > 0 else 0

        # Compute per_stream from streaming data only
        streaming_by_type = self._streaming_df.groupby('release_type').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum',
        }).reset_index()
        streaming_by_type.columns = ['release_type', 'str_streams', 'str_earnings']
        streaming_by_type['per_stream'] = (streaming_by_type['str_earnings'] / streaming_by_type['str_streams']).round(4)

        summary = summary.merge(streaming_by_type[['release_type', 'per_stream']], on='release_type', how='left')
        summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0)
        summary = summary.sort_values('earnings', ascending=False)

        return {
            "available": True,
            "summary": summary.to_dict('records'),
        }

    def get_release_type_monthly_trend(self) -> Dict[str, Any]:
        """Monthly earnings trend split by release type."""
        if not self._has_upc:
            return {"available": False, "reason": "No UPC column found in data"}

        monthly = self.df.groupby([
            self.df['Sale Month'].dt.strftime('%Y-%m'),
            'release_type'
        ]).agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum',
        }).reset_index()

        monthly.columns = ['month', 'release_type', 'streams', 'earnings']
        monthly['earnings'] = monthly['earnings'].round(2)
        monthly = monthly.sort_values(['month', 'release_type'])

        return {
            "available": True,
            "monthly": monthly.to_dict('records'),
        }

    def get_overview(self) -> Dict[str, Any]:
        """Get high-level summary statistics"""
        total_earnings = float(self.df['Earnings (USD)'].sum())
        total_streams = int(self.df['Quantity'].sum())

        streaming_earnings = float(self._streaming_df['Earnings (USD)'].sum())
        streaming_streams = int(self._streaming_df['Quantity'].sum())
        download_earnings = float(self._purchase_df['Earnings (USD)'].sum())
        download_units = int(self._purchase_df['Quantity'].sum())

        return {
            "total_earnings": round(total_earnings, 2),
            "total_streams": total_streams,
            "unique_songs": int(self.df['Title'].nunique()),
            "unique_platforms": int(self.df['Store'].nunique()),
            "unique_countries": int(self.df['Country of Sale'].nunique()),
            "avg_per_stream": round(streaming_earnings / streaming_streams, 4) if streaming_streams > 0 else 0,
            "streaming_earnings": round(streaming_earnings, 2),
            "download_earnings": round(download_earnings, 2),
            "download_units": download_units,
            "non_streaming_platforms": sorted(self._non_streaming),
            "date_range": {
                "start": self.df['Sale Month'].min().strftime('%Y-%m'),
                "end": self.df['Sale Month'].max().strftime('%Y-%m')
            }
        }
    
    def get_monthly_trend(self) -> List[Dict]:
        """Get earnings and streams by month"""
        monthly = self.df.groupby('Sale Month').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        
        monthly.columns = ['month', 'streams', 'earnings']
        monthly['month'] = monthly['month'].dt.strftime('%Y-%m')
        monthly['earnings'] = monthly['earnings'].round(2)
        monthly = monthly.sort_values('month')
        
        # Calculate month-over-month change
        monthly['earnings_change_pct'] = monthly['earnings'].pct_change() * 100
        monthly['earnings_change_pct'] = monthly['earnings_change_pct'].round(1).fillna(0)
        
        return monthly.to_dict('records')
    
    def get_yearly_trend(self) -> List[Dict]:
        """Get earnings and streams by year"""
        yearly = self.df.groupby('Year').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        
        yearly.columns = ['year', 'streams', 'earnings']
        yearly['earnings'] = yearly['earnings'].round(2)
        
        # Calculate YoY change
        yearly['yoy_change_pct'] = yearly['earnings'].pct_change() * 100
        yearly['yoy_change_pct'] = yearly['yoy_change_pct'].round(1).fillna(0)
        
        return yearly.to_dict('records')
    
    def get_song_breakdown(self, limit: int = 20) -> List[Dict]:
        """Get earnings breakdown by song"""
        total_earnings = self.df['Earnings (USD)'].sum()

        songs = self.df.groupby('Title').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        songs.columns = ['title', 'streams', 'earnings']

        # Compute per_stream from streaming data only
        streaming_songs = self._streaming_df.groupby('Title').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        streaming_songs.columns = ['title', 'streaming_streams', 'streaming_earnings']
        streaming_songs['per_stream'] = (streaming_songs['streaming_earnings'] / streaming_songs['streaming_streams']).round(4)

        songs = songs.merge(streaming_songs[['title', 'per_stream']], on='title', how='left')
        songs['per_stream'] = songs['per_stream'].fillna(0)
        songs['pct_of_total'] = (songs['earnings'] / total_earnings * 100).round(1)
        songs['earnings'] = songs['earnings'].round(2)
        songs = songs.sort_values('earnings', ascending=False)

        return songs.head(limit).to_dict('records')
    
    def get_platform_breakdown(self, limit: int = 15) -> List[Dict]:
        """Get earnings breakdown by platform/store"""
        total_earnings = self.df['Earnings (USD)'].sum()

        platforms = self.df.groupby('Store').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()

        platforms.columns = ['platform', 'streams', 'earnings']
        platforms['is_streaming'] = ~platforms['platform'].isin(self._non_streaming)
        streaming_mask = platforms['is_streaming']
        platforms['per_stream'] = None
        if streaming_mask.any():
            platforms.loc[streaming_mask, 'per_stream'] = (
                platforms.loc[streaming_mask, 'earnings'] / platforms.loc[streaming_mask, 'streams']
            ).round(4)
        platforms['pct_of_total'] = (platforms['earnings'] / total_earnings * 100).round(1)
        platforms['earnings'] = platforms['earnings'].round(2)
        platforms = platforms.sort_values('earnings', ascending=False)

        return platforms.head(limit).to_dict('records')
    
    def get_country_breakdown(self, limit: int = 20) -> List[Dict]:
        """Get earnings breakdown by country"""
        total_earnings = self.df['Earnings (USD)'].sum()

        countries = self.df.groupby('Country of Sale').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        countries.columns = ['country', 'streams', 'earnings']

        # Compute per_stream from streaming data only
        streaming_countries = self._streaming_df.groupby('Country of Sale').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        streaming_countries.columns = ['country', 'streaming_streams', 'streaming_earnings']
        streaming_countries['per_stream'] = (streaming_countries['streaming_earnings'] / streaming_countries['streaming_streams']).round(4)

        countries = countries.merge(streaming_countries[['country', 'per_stream']], on='country', how='left')
        countries['per_stream'] = countries['per_stream'].fillna(0)
        countries['pct_of_total'] = (countries['earnings'] / total_earnings * 100).round(1)
        countries['earnings'] = countries['earnings'].round(2)
        countries = countries.sort_values('earnings', ascending=False)

        return countries.head(limit).to_dict('records')
    
    def get_concentration_metrics(self) -> Dict[str, Any]:
        """Calculate concentration risk metrics"""
        total = self.df['Earnings (USD)'].sum()
        
        songs = self.df.groupby('Title')['Earnings (USD)'].sum().sort_values(ascending=False)
        platforms = self.df.groupby('Store')['Earnings (USD)'].sum().sort_values(ascending=False)
        countries = self.df.groupby('Country of Sale')['Earnings (USD)'].sum().sort_values(ascending=False)
        
        return {
            "song": {
                "top_1_pct": round(float(songs.iloc[0] / total * 100), 1),
                "top_1_name": songs.index[0],
                "top_3_pct": round(float(songs.head(3).sum() / total * 100), 1),
                "top_10_pct": round(float(songs.head(10).sum() / total * 100), 1),
            },
            "platform": {
                "top_1_pct": round(float(platforms.iloc[0] / total * 100), 1),
                "top_1_name": platforms.index[0],
                "top_3_pct": round(float(platforms.head(3).sum() / total * 100), 1),
            },
            "country": {
                "top_1_pct": round(float(countries.iloc[0] / total * 100), 1),
                "top_1_name": countries.index[0],
                "top_5_pct": round(float(countries.head(5).sum() / total * 100), 1),
            },
            "underperforming_songs": int(len(songs[songs / total < 0.01]))  # Songs with <1% of revenue
        }
    
    def get_growth_analysis(self, months: int = 3) -> Dict[str, Any]:
        """Compare recent period vs prior period"""
        max_date = self.df['Sale Month'].max()
        
        # Recent period
        recent_start = max_date - pd.DateOffset(months=months)
        recent_df = self.df[self.df['Sale Month'] > recent_start]
        
        # Prior period
        prior_start = recent_start - pd.DateOffset(months=months)
        prior_df = self.df[(self.df['Sale Month'] > prior_start) & (self.df['Sale Month'] <= recent_start)]
        
        # Song-level comparison
        recent_by_song = recent_df.groupby('Title')['Earnings (USD)'].sum()
        prior_by_song = prior_df.groupby('Title')['Earnings (USD)'].sum()
        
        growth = pd.DataFrame({
            'recent': recent_by_song,
            'previous': prior_by_song
        }).fillna(0)
        
        growth['change'] = growth['recent'] - growth['previous']
        growth['change_pct'] = ((growth['recent'] / growth['previous']) - 1) * 100
        growth = growth.replace([np.inf, -np.inf], np.nan)
        
        growing = growth[growth['change'] > 0].sort_values('change', ascending=False).head(5)
        declining = growth[growth['change'] < 0].sort_values('change', ascending=True).head(5)
        
        return {
            "period": f"Last {months} months vs prior {months} months",
            "growing": [
                {
                    "title": title,
                    "change": round(row['change'], 2),
                    "change_pct": round(row['change_pct'], 1) if not pd.isna(row['change_pct']) else None
                }
                for title, row in growing.iterrows()
            ],
            "declining": [
                {
                    "title": title,
                    "change": round(row['change'], 2),
                    "change_pct": round(row['change_pct'], 1) if not pd.isna(row['change_pct']) else None
                }
                for title, row in declining.iterrows()
            ]
        }
    
    def get_platform_song_matrix(self, top_songs: int = 5) -> List[Dict]:
        """Get platform breakdown for top songs"""
        top_song_titles = self.df.groupby('Title')['Earnings (USD)'].sum().sort_values(ascending=False).head(top_songs).index

        results = []
        for song in top_song_titles:
            song_df = self.df[self.df['Title'] == song]
            platforms = song_df.groupby('Store').agg({
                'Quantity': 'sum',
                'Earnings (USD)': 'sum'
            })
            is_streaming = ~platforms.index.isin(self._non_streaming)
            platforms['per_stream'] = np.where(
                is_streaming,
                (platforms['Earnings (USD)'] / platforms['Quantity']).round(4),
                np.nan
            )
            platforms['pct'] = platforms['Earnings (USD)'] / platforms['Earnings (USD)'].sum() * 100
            platforms = platforms.sort_values('Earnings (USD)', ascending=False)

            results.append({
                "song": song,
                "platforms": [
                    {
                        "platform": platform,
                        "earnings": round(row['Earnings (USD)'], 2),
                        "per_stream": round(row['per_stream'], 4) if not pd.isna(row['per_stream']) else None,
                        "pct": round(row['pct'], 1)
                    }
                    for platform, row in platforms.head(5).iterrows()
                ]
            })

        return results
    
    def get_high_value_markets(self) -> Dict[str, List[Dict]]:
        """Identify high-value vs high-volume markets (streaming only)"""
        countries = self._streaming_df.groupby('Country of Sale').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        })
        countries['per_stream'] = countries['Earnings (USD)'] / countries['Quantity']

        # High value = high $/stream, decent volume
        high_value = countries[countries['Quantity'] > 10000].sort_values('per_stream', ascending=False).head(5)

        # High volume, low value
        high_volume_low_value = countries[countries['Quantity'] > 100000].sort_values('per_stream', ascending=True).head(5)

        return {
            "high_value": [
                {
                    "country": country,
                    "per_stream": round(row['per_stream'], 4),
                    "earnings": round(row['Earnings (USD)'], 2),
                    "streams": int(row['Quantity'])
                }
                for country, row in high_value.iterrows()
            ],
            "high_volume_low_value": [
                {
                    "country": country,
                    "per_stream": round(row['per_stream'], 4),
                    "earnings": round(row['Earnings (USD)'], 2),
                    "streams": int(row['Quantity'])
                }
                for country, row in high_volume_low_value.iterrows()
            ]
        }
    
    def get_catalog_excluding_top_song(self) -> Dict[str, Any]:
        """Analyze catalog performance excluding the #1 song"""
        top_song = self.df.groupby('Title')['Earnings (USD)'].sum().idxmax()

        # Check for variants (e.g., "The Unknowing", "The Unknowing (Slowed)")
        base_name = top_song.split('(')[0].strip()
        df_filtered = self.df[~self.df['Title'].str.contains(base_name, case=False, na=False)]
        streaming_filtered = self._streaming_df[~self._streaming_df['Title'].str.contains(base_name, case=False, na=False)]

        total_filtered = df_filtered['Earnings (USD)'].sum()

        songs = df_filtered.groupby('Title').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        songs.columns = ['title', 'streams', 'earnings']

        # per_stream from streaming data only
        streaming_songs = streaming_filtered.groupby('Title').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        streaming_songs.columns = ['title', 'str_streams', 'str_earnings']
        streaming_songs['per_stream'] = (streaming_songs['str_earnings'] / streaming_songs['str_streams']).round(4)

        songs = songs.merge(streaming_songs[['title', 'per_stream']], on='title', how='left')
        songs['per_stream'] = songs['per_stream'].fillna(0)
        songs['earnings'] = songs['earnings'].round(2)
        songs = songs.sort_values('earnings', ascending=False)

        return {
            "excluded_song": top_song,
            "excluded_pattern": base_name,
            "remaining_earnings": round(total_filtered, 2),
            "top_songs": songs.head(10).to_dict('records')
        }
    
    def get_monthly_breakdown(self) -> List[Dict]:
        """
        Get GRANULAR monthly breakdown by song, platform, and country.

        This is the source-of-truth data that enables accurate client-side filtering.
        Each row represents a unique (month, song, platform, country) combination.

        Returns a list of dicts with:
            - month: YYYY-MM format
            - song: Song title
            - platform: Store/platform name
            - country: Country code
            - earnings: Total earnings for this combination
            - gross_earnings: Earnings before splits
            - team_percentage: Artist's ownership percentage
            - streams: Total streams for this combination
        """
        # Create a month string column for grouping (YYYY-MM format)
        df_copy = self.df.copy()
        df_copy['month_str'] = df_copy['Sale Month'].dt.strftime('%Y-%m')

        # Group by all dimensions (include release fields and team_percentage if available)
        group_cols = ['month_str', 'Title', 'Store', 'Country of Sale']
        if self._has_upc:
            group_cols.extend(['release_type', 'release_name'])
        # team_percentage is per-song so it's safe to group by
        group_cols.append('team_percentage')

        agg_dict = {
            'Quantity': 'sum',
            'Earnings (USD)': 'sum',
            'gross_earnings': 'sum',
        }

        breakdown = df_copy.groupby(group_cols).agg(agg_dict).reset_index()

        # Build column names dynamically
        base_cols = ['month', 'song', 'platform', 'country']
        if self._has_upc:
            base_cols.extend(['release_type', 'release_name'])
        base_cols.extend(['team_percentage', 'streams', 'earnings', 'gross_earnings'])
        breakdown.columns = base_cols

        # Round earnings to 2 decimal places
        breakdown['earnings'] = breakdown['earnings'].round(2)
        breakdown['gross_earnings'] = breakdown['gross_earnings'].round(2)

        # Convert streams to int
        breakdown['streams'] = breakdown['streams'].astype(int)

        # Sort by month, then earnings descending
        breakdown = breakdown.sort_values(['month', 'earnings'], ascending=[True, False])

        return breakdown.to_dict('records')
    
    # ------------------------------------------------------------------
    # Ownership & Release Strategy Analysis
    # ------------------------------------------------------------------

    def get_ownership_summary(self) -> Dict[str, Any]:
        """Simple ownership breakdown: full vs partial, gross vs net."""
        if not self._has_team_pct:
            return {"available": False, "reason": "No Team Percentage column in data"}

        full_mask = self.df['team_percentage'] >= 100
        partial_mask = ~full_mask

        your_total = float(self.df['Earnings (USD)'].sum())
        gross_total = float(self.df['gross_earnings'].sum())

        return {
            "available": True,
            "full_ownership_songs": int(self.df.loc[full_mask, 'Title'].nunique()),
            "partial_ownership_songs": int(self.df.loc[partial_mask, 'Title'].nunique()),
            "full_ownership_earnings": round(float(self.df.loc[full_mask, 'Earnings (USD)'].sum()), 2),
            "partial_ownership_earnings": round(float(self.df.loc[partial_mask, 'Earnings (USD)'].sum()), 2),
            "gross_revenue_generated": round(gross_total, 2),
            "your_total_earnings": round(your_total, 2),
            "partner_share": round(gross_total - your_total, 2),
            "effective_catalog_ownership_pct": round((your_total / gross_total * 100) if gross_total > 0 else 100.0, 1),
        }

    def get_release_strategy_analysis(self) -> Dict[str, Any]:
        """Forward-looking release strategy: type performance, cross-release
        cannibalization, platform affinity by type, and recommendations."""
        if not self._has_upc:
            return {"available": False, "reason": "No UPC/release data available"}

        sdf = self._streaming_df
        result: Dict[str, Any] = {"available": True}

        # ---------------------------------------------------------------
        # 1. release_type_performance
        # ---------------------------------------------------------------
        total_earnings = float(sdf['Earnings (USD)'].sum())
        type_perf = {}
        for rtype in ['Single', 'EP', 'Album']:
            rt_data = sdf[sdf['release_type'] == rtype]
            if rt_data.empty:
                continue
            streams = int(rt_data['Quantity'].sum())
            earnings = float(rt_data['Earnings (USD)'].sum())
            per_stream = round(earnings / streams, 4) if streams > 0 else 0
            pct = round(earnings / total_earnings * 100, 1) if total_earnings > 0 else 0
            type_perf[rtype.lower() + 's'] = {
                "count": int(rt_data['UPC'].nunique()),
                "total_streams": streams,
                "total_earnings": round(earnings, 2),
                "per_stream": per_stream,
                "pct_of_catalog_earnings": pct,
            }

        # Generate comparison insight
        rates = {k: v['per_stream'] for k, v in type_perf.items() if v['per_stream'] > 0}
        if len(rates) >= 2:
            best_type = max(rates, key=rates.get)
            worst_type = min(rates, key=rates.get)
            if rates[best_type] > 0 and rates[worst_type] > 0:
                pct_diff = round((rates[best_type] / rates[worst_type] - 1) * 100, 0)
                type_perf['insight'] = f"{best_type.title()} earn {pct_diff:.0f}% more per stream than {worst_type}"
            else:
                type_perf['insight'] = None
        else:
            type_perf['insight'] = None

        result['release_type_performance'] = type_perf

        # ---------------------------------------------------------------
        # 2. cross_release_performance
        # ---------------------------------------------------------------
        # Find songs that appear on multiple release types (Single AND Album/EP)
        has_release_type = 'release_type' in sdf.columns
        cross_songs = []
        avg_single_capture = []

        if has_release_type:
            song_types = sdf.groupby('Title')['release_type'].nunique()
            multi_type_songs = song_types[song_types > 1].index

            for title in multi_type_songs:
                song_data = sdf[sdf['Title'] == title]
                by_type = song_data.groupby('release_type').agg({
                    'Quantity': 'sum',
                    'Earnings (USD)': 'sum',
                }).reset_index()

                total_s = int(by_type['Quantity'].sum())
                if total_s == 0:
                    continue

                single_row = by_type[by_type['release_type'] == 'Single']
                album_row = by_type[by_type['release_type'].isin(['Album', 'EP'])]

                single_streams = int(single_row['Quantity'].sum()) if not single_row.empty else 0
                single_earnings = float(single_row['Earnings (USD)'].sum()) if not single_row.empty else 0
                album_streams = int(album_row['Quantity'].sum()) if not album_row.empty else 0
                album_earnings = float(album_row['Earnings (USD)'].sum()) if not album_row.empty else 0

                single_capture = round(single_streams / total_s * 100, 0) if total_s > 0 else 0

                entry: Dict[str, Any] = {
                    "title": title,
                    "single_streams": single_streams,
                    "single_earnings": round(single_earnings, 2),
                    "single_per_stream": round(single_earnings / single_streams, 4) if single_streams > 0 else 0,
                    "album_streams": album_streams,
                    "album_earnings": round(album_earnings, 2),
                    "album_per_stream": round(album_earnings / album_streams, 4) if album_streams > 0 else 0,
                    "single_capture_pct": single_capture,
                }

                if single_capture > 75:
                    entry['insight'] = "Single version dominates"
                elif single_capture > 50:
                    entry['insight'] = "Single leads but album contributes"
                elif single_capture > 0:
                    entry['insight'] = "Album version outperforms single"
                else:
                    entry['insight'] = "Only on album/EP"

                cross_songs.append(entry)
                if single_streams > 0:
                    avg_single_capture.append(single_capture)

            cross_songs.sort(key=lambda x: x['single_earnings'] + x['album_earnings'], reverse=True)

        avg_capture = round(float(np.mean(avg_single_capture)), 0) if avg_single_capture else 0
        summary_text = (
            f"Singles capture {avg_capture:.0f}% of streams when a song appears on both single and album"
            if avg_single_capture
            else "No cross-released songs found"
        )

        result['cross_release_performance'] = {
            "songs": cross_songs,
            "average_single_capture_pct": avg_capture,
            "summary": summary_text,
        }

        # ---------------------------------------------------------------
        # 3. platform_by_release_type
        # ---------------------------------------------------------------
        plat_by_type = []
        if has_release_type:
            for rtype in ['Single', 'EP', 'Album']:
                rt_data = sdf[sdf['release_type'] == rtype]
                if rt_data.empty:
                    continue
                plat_agg = rt_data.groupby('Store').agg({
                    'Quantity': 'sum', 'Earnings (USD)': 'sum'
                }).reset_index()
                plat_agg = plat_agg[plat_agg['Quantity'] >= 1000]
                if plat_agg.empty:
                    continue

                top_plat = plat_agg.sort_values('Earnings (USD)', ascending=False).iloc[0]
                rt_total = rt_data['Earnings (USD)'].sum()

                plat_by_type.append({
                    "release_type": rtype,
                    "top_platform": top_plat['Store'],
                    "per_stream": round(float(top_plat['Earnings (USD)'] / top_plat['Quantity']), 4) if top_plat['Quantity'] > 0 else 0,
                    "pct_of_type_earnings": round(float(top_plat['Earnings (USD)'] / rt_total * 100), 1) if rt_total > 0 else 0,
                })

        result['platform_by_release_type'] = plat_by_type

        # ---------------------------------------------------------------
        # 4. release_strategy_recommendations
        # ---------------------------------------------------------------
        recs = []
        tp = result['release_type_performance']
        crp = result['cross_release_performance']

        # Recommendation: release singles first
        if avg_capture > 70:
            recs.append({
                "priority": "high",
                "recommendation": "Release singles first",
                "reason": f"Singles capture {avg_capture:.0f}% of total streams when a song is on both",
                "impact": "Maximize early revenue before album release",
            })

        # Recommendation: singles earn more per stream
        singles_rate = tp.get('singles', {}).get('per_stream', 0)
        albums_rate = tp.get('albums', {}).get('per_stream', 0)
        eps_rate = tp.get('eps', {}).get('per_stream', 0)

        if singles_rate > 0 and albums_rate > 0 and singles_rate > albums_rate:
            pct_more = round((singles_rate / albums_rate - 1) * 100, 0)
            recs.append({
                "priority": "high",
                "recommendation": "Focus campaigns on singles",
                "reason": f"Singles earn ${singles_rate:.4f}/stream vs ${albums_rate:.4f} on albums (+{pct_more:.0f}%)",
                "impact": "Better ROI on promotion spend",
            })

        # Recommendation: album role
        album_data = tp.get('albums', {})
        if album_data:
            album_pct = album_data.get('pct_of_catalog_earnings', 0)
            album_count = album_data.get('count', 0)
            if album_pct > 0 and album_pct < 30 and album_count > 0:
                recs.append({
                    "priority": "medium",
                    "recommendation": "Use albums for catalog depth, not primary revenue",
                    "reason": f"Album tracks earn {album_pct}% of revenue",
                    "impact": "Set realistic expectations for album performance",
                })
            elif album_pct >= 30:
                recs.append({
                    "priority": "medium",
                    "recommendation": "Albums are a strong revenue driver — keep releasing them",
                    "reason": f"Album tracks earn {album_pct}% of your catalog revenue",
                    "impact": "Continue investing in full-length releases",
                })

        # Recommendation: EP performance
        if eps_rate > 0 and singles_rate > 0:
            if eps_rate > singles_rate * 0.95:
                recs.append({
                    "priority": "medium",
                    "recommendation": "EPs perform nearly as well as singles per stream",
                    "reason": f"EP rate ${eps_rate:.4f} vs single rate ${singles_rate:.4f}",
                    "impact": "EPs can be a good middle ground for releasing multiple tracks",
                })
            elif eps_rate < singles_rate * 0.8:
                ep_drop = round((1 - eps_rate / singles_rate) * 100, 0)
                recs.append({
                    "priority": "low",
                    "recommendation": "Consider releasing EP tracks as singles instead",
                    "reason": f"EP tracks earn {ep_drop:.0f}% less per stream than singles",
                    "impact": "Individual singles may generate more revenue",
                })

        # Recommendation: platform focus by type
        if plat_by_type:
            for pbt in plat_by_type:
                if pbt['pct_of_type_earnings'] > 40:
                    recs.append({
                        "priority": "low",
                        "recommendation": f"Prioritize {pbt['top_platform']} for {pbt['release_type'].lower()} releases",
                        "reason": f"{pbt['top_platform']} drives {pbt['pct_of_type_earnings']}% of {pbt['release_type'].lower()} earnings",
                        "impact": f"Focus promotion spend on the highest-performing platform for this format",
                    })

        result['release_strategy_recommendations'] = recs

        return result

    def get_full_analysis(self) -> Dict[str, Any]:
        """Get complete analysis - all metrics including granular breakdown"""
        result = {
            "overview": self.get_overview(),
            "monthly_trend": self.get_monthly_trend(),
            "yearly_trend": self.get_yearly_trend(),
            "songs": self.get_song_breakdown(),
            "platforms": self.get_platform_breakdown(),
            "countries": self.get_country_breakdown(),
            "concentration": self.get_concentration_metrics(),
            "growth": self.get_growth_analysis(),
            "platform_song_matrix": self.get_platform_song_matrix(),
            "market_analysis": self.get_high_value_markets(),
            "catalog_depth": self.get_catalog_excluding_top_song(),
            # Granular data for client-side filtering
            "monthly_breakdown": self.get_monthly_breakdown(),
            # Release classification (Singles / EPs / Albums via UPC)
            "release_breakdown": self.get_release_breakdown(),
            "release_type_summary": self.get_release_type_summary(),
            "release_type_monthly_trend": self.get_release_type_monthly_trend(),
            # Ownership & release strategy
            "ownership_summary": self.get_ownership_summary(),
            "release_strategy_analysis": self.get_release_strategy_analysis(),
        }

        # Deep insights (ML-driven) — graceful degradation if libs missing
        try:
            from .insights_engine import InsightsEngine
            engine = InsightsEngine(self.df, self._streaming_df, self._non_streaming, self._has_upc)
            result["deep_insights"] = engine.get_all_insights()
        except ImportError:
            result["deep_insights"] = {"available": False, "reason": "ML libraries not installed"}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Deep insights failed: {e}", exc_info=True)
            result["deep_insights"] = {"available": False, "error": str(e)}

        return result


# For testing
if __name__ == "__main__":
    # Test with sample data
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, sep='\t')
        
        analyzer = DistroKidAnalyzer(df)
        results = analyzer.get_full_analysis()
        
        import json
        print(json.dumps(results, indent=2))
        
        # Print breakdown stats
        breakdown = results.get('monthly_breakdown', [])
        print(f"\n--- Monthly Breakdown Stats ---")
        print(f"Total rows: {len(breakdown)}")
        if breakdown:
            print(f"Sample row: {breakdown[0]}")
    else:
        print("Usage: python analytics_engine.py <path_to_distrokid_export>")
