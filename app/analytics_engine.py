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
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
    
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
    
    def get_overview(self) -> Dict[str, Any]:
        """Get high-level summary statistics"""
        total_earnings = float(self.df['Earnings (USD)'].sum())
        total_streams = int(self.df['Quantity'].sum())
        
        return {
            "total_earnings": round(total_earnings, 2),
            "total_streams": total_streams,
            "unique_songs": int(self.df['Title'].nunique()),
            "unique_platforms": int(self.df['Store'].nunique()),
            "unique_countries": int(self.df['Country of Sale'].nunique()),
            "avg_per_stream": round(total_earnings / total_streams, 4) if total_streams > 0 else 0,
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
        songs['per_stream'] = (songs['earnings'] / songs['streams']).round(4)
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
        platforms['per_stream'] = (platforms['earnings'] / platforms['streams']).round(4)
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
        countries['per_stream'] = (countries['earnings'] / countries['streams']).round(4)
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
            platforms['per_stream'] = platforms['Earnings (USD)'] / platforms['Quantity']
            platforms['pct'] = platforms['Earnings (USD)'] / platforms['Earnings (USD)'].sum() * 100
            platforms = platforms.sort_values('Earnings (USD)', ascending=False)
            
            results.append({
                "song": song,
                "platforms": [
                    {
                        "platform": platform,
                        "earnings": round(row['Earnings (USD)'], 2),
                        "per_stream": round(row['per_stream'], 4),
                        "pct": round(row['pct'], 1)
                    }
                    for platform, row in platforms.head(5).iterrows()
                ]
            })
        
        return results
    
    def get_high_value_markets(self) -> Dict[str, List[Dict]]:
        """Identify high-value vs high-volume markets"""
        countries = self.df.groupby('Country of Sale').agg({
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
        
        total_filtered = df_filtered['Earnings (USD)'].sum()
        
        songs = df_filtered.groupby('Title').agg({
            'Quantity': 'sum',
            'Earnings (USD)': 'sum'
        }).reset_index()
        songs.columns = ['title', 'streams', 'earnings']
        songs['per_stream'] = (songs['earnings'] / songs['streams']).round(4)
        songs['earnings'] = songs['earnings'].round(2)
        songs = songs.sort_values('earnings', ascending=False)
        
        return {
            "excluded_song": top_song,
            "excluded_pattern": base_name,
            "remaining_earnings": round(total_filtered, 2),
            "top_songs": songs.head(10).to_dict('records')
        }
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """Get complete analysis - all metrics"""
        return {
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
            "catalog_depth": self.get_catalog_excluding_top_song()
        }


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
    else:
        print("Usage: python analytics_engine.py <path_to_distrokid_export>")
