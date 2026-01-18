"""
Alternative Data Integration Module
Sentiment analysis, ESG factors, economic indicators
Beyond traditional bond data
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class AlternativeDataAnalyzer:
    """
    Alternative data analysis for bond pricing
    Integrates sentiment, ESG, economic factors
    More comprehensive than traditional approaches
    """
    
    def __init__(self, valuator: BondValuator = None):
        """Initialize alternative data analyzer"""
        self.valuator = valuator if valuator else BondValuator()
    
    def calculate_esg_score(self, bond: Bond) -> Dict:
        """
        Calculate ESG (Environmental, Social, Governance) score
        
        ESG factors increasingly important in bond valuation
        
        Args:
            bond: Bond object
            
        Returns:
            ESG score and impact on valuation
        """
        # Simulate ESG scoring based on issuer characteristics
        # In production, would use actual ESG data (MSCI, Sustainalytics)
        
        issuer = bond.issuer.lower()
        
        # Environmental score (0-100)
        if 'energy' in issuer or 'oil' in issuer:
            env_score = 40  # Lower for energy
        elif 'tech' in issuer or 'renewable' in issuer:
            env_score = 80  # Higher for tech
        else:
            env_score = 60  # Neutral
        
        # Social score
        if 'healthcare' in issuer or 'education' in issuer:
            social_score = 75
        else:
            social_score = 60
        
        # Governance score (based on credit rating)
        rating_map = {'AAA': 90, 'AA': 85, 'A': 75, 'BBB': 65, 'BB': 50, 'B': 35, 'CCC': 20}
        gov_score = rating_map.get(bond.credit_rating.upper(), 60)
        
        # Overall ESG score (weighted average)
        esg_score = (0.4 * env_score + 0.3 * social_score + 0.3 * gov_score)
        
        # ESG impact on spread (better ESG = tighter spread)
        # In production, would use empirical ESG-spread relationship
        esg_spread_adjustment = (100 - esg_score) / 100 * 0.001  # Max 10bp adjustment
        
        # Adjusted fair value with ESG
        base_fair_value = self.valuator.calculate_fair_value(bond)
        ytm = self.valuator.calculate_yield_to_maturity(bond)
        
        # ESG-adjusted YTM (better ESG = lower required yield)
        esg_adjusted_ytm = ytm - esg_spread_adjustment
        
        # Approximate price adjustment using duration
        duration = self.valuator.calculate_duration(bond, ytm)
        price_impact = duration * esg_spread_adjustment
        esg_adjusted_value = base_fair_value * (1 + price_impact)
        
        return {
            'esg_score': esg_score,
            'environmental_score': env_score,
            'social_score': social_score,
            'governance_score': gov_score,
            'esg_spread_adjustment_bps': esg_spread_adjustment * 10000,
            'base_fair_value': base_fair_value,
            'esg_adjusted_value': esg_adjusted_value,
            'esg_impact_pct': price_impact * 100,
            'esg_rating': self._esg_rating(esg_score)
        }
    
    def _esg_rating(self, score: float) -> str:
        """Convert ESG score to rating"""
        if score >= 80:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 60:
            return 'Average'
        elif score >= 50:
            return 'Below Average'
        else:
            return 'Poor'
    
    def sentiment_analysis(self, bond: Bond, news_score: Optional[float] = None) -> Dict:
        """
        Sentiment analysis impact on bond pricing
        
        News sentiment can affect bond prices
        
        Args:
            bond: Bond object
            news_score: News sentiment score (-1 to 1, negative = bad, positive = good)
            
        Returns:
            Sentiment impact analysis
        """
        if news_score is None:
            # Simulate sentiment (in production, would use NLP on news articles)
            news_score = np.random.uniform(-0.5, 0.5)
        
        # Sentiment impact on credit spread
        # Positive sentiment = tighter spread, negative = wider spread
        sentiment_spread_adjustment = -news_score * 0.002  # Max 20bp impact
        
        base_fair_value = self.valuator.calculate_fair_value(bond)
        ytm = self.valuator.calculate_yield_to_maturity(bond)
        duration = self.valuator.calculate_duration(bond, ytm)
        
        # Price impact
        price_impact = duration * sentiment_spread_adjustment
        sentiment_adjusted_value = base_fair_value * (1 + price_impact)
        
        return {
            'news_sentiment': news_score,
            'sentiment_label': 'Positive' if news_score > 0 else 'Negative' if news_score < 0 else 'Neutral',
            'sentiment_spread_adjustment_bps': sentiment_spread_adjustment * 10000,
            'base_fair_value': base_fair_value,
            'sentiment_adjusted_value': sentiment_adjusted_value,
            'sentiment_impact_pct': price_impact * 100
        }
    
    def economic_factors_impact(
        self,
        bonds: List[Bond],
        inflation_expectation: float = 0.02,
        gdp_growth: float = 0.03,
        unemployment: float = 0.04
    ) -> Dict:
        """
        Analyze impact of economic factors on bond portfolio
        
        Macro factors affect bond prices
        
        Args:
            bonds: List of bonds
            inflation_expectation: Expected inflation rate
            gdp_growth: GDP growth rate
            unemployment: Unemployment rate
            
        Returns:
            Economic factor impact analysis
        """
        impacts = []
        
        for bond in bonds:
            # Inflation impact: hurts long-duration bonds more
            duration = self.valuator.calculate_duration(bond)
            inflation_impact = -duration * inflation_expectation
            
            # GDP impact: better growth = higher yields = lower prices
            gdp_impact = -duration * gdp_growth * 0.5  # 50% pass-through
            
            # Unemployment impact: higher unemployment = lower yields = higher prices
            unemployment_impact = duration * unemployment * 0.3  # 30% pass-through
            
            total_impact = inflation_impact + gdp_impact + unemployment_impact
            
            base_value = self.valuator.calculate_fair_value(bond)
            adjusted_value = base_value * (1 + total_impact)
            
            impacts.append({
                'bond_id': bond.bond_id,
                'inflation_impact_pct': inflation_impact * 100,
                'gdp_impact_pct': gdp_impact * 100,
                'unemployment_impact_pct': unemployment_impact * 100,
                'total_impact_pct': total_impact * 100,
                'adjusted_value': adjusted_value,
                'base_value': base_value
            })
        
        avg_impact = np.mean([i['total_impact_pct'] for i in impacts])
        
        return {
            'bond_impacts': impacts,
            'average_impact_pct': avg_impact,
            'inflation_expectation': inflation_expectation,
            'gdp_growth': gdp_growth,
            'unemployment': unemployment
        }
    
    def macro_factor_adjusted_valuation(
        self,
        bond: Bond,
        inflation: float = 0.02,
        gdp: float = 0.03,
        sentiment: float = 0.0
    ) -> Dict:
        """
        Comprehensive valuation adjusted for macro factors
        
        Combines ESG, sentiment, and economic factors
        
        Args:
            bond: Bond object
            inflation: Inflation expectation
            gdp: GDP growth
            sentiment: News sentiment
            
        Returns:
            Macro-adjusted valuation
        """
        # Base valuation
        base_value = self.valuator.calculate_fair_value(bond)
        
        # ESG impact
        esg_result = self.calculate_esg_score(bond)
        esg_impact = (esg_result['esg_adjusted_value'] - base_value) / base_value
        
        # Sentiment impact
        sentiment_result = self.sentiment_analysis(bond, sentiment)
        sentiment_impact = (sentiment_result['sentiment_adjusted_value'] - base_value) / base_value
        
        # Economic impact
        duration = self.valuator.calculate_duration(bond)
        inflation_impact = -duration * inflation
        gdp_impact = -duration * gdp * 0.5
        
        # Total adjustment
        total_impact = esg_impact + sentiment_impact + inflation_impact + gdp_impact
        macro_adjusted_value = base_value * (1 + total_impact)
        
        return {
            'base_fair_value': base_value,
            'macro_adjusted_value': macro_adjusted_value,
            'total_adjustment_pct': total_impact * 100,
            'esg_impact_pct': esg_impact * 100,
            'sentiment_impact_pct': sentiment_impact * 100,
            'economic_impact_pct': (inflation_impact + gdp_impact) * 100,
            'components': {
                'esg_score': esg_result['esg_score'],
                'sentiment': sentiment,
                'inflation': inflation,
                'gdp_growth': gdp
            }
        }
