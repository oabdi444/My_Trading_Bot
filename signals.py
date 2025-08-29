import pandas as pd

class SignalEngine:
    def __init__(self, signal_weights, confidence_threshold, min_confluence):
        self.signal_weights = signal_weights
        self.confidence_threshold = confidence_threshold
        self.min_confluence = min_confluence

    def generate_enhanced_signal(self, df, latest, prev, market_condition, mlp, sentiment, tf_consensus):
        reasons = []
        # Trend
        trend_score = self.trend_following_signal(df, latest, prev, reasons)
        mean_rev = self.mean_reversion_signal(df, latest, prev, reasons)
        momentum = self.momentum_signal(df, latest, prev, reasons)
        volume = self.volume_analysis_signal(df, latest, prev, reasons)
        volatility = self.volatility_signal(df, latest, prev, reasons)
        ml_pred = self.ml_prediction_signal(mlp, reasons)

        # Weighted sum
        confluence = sum([
            abs(trend_score) > 0,
            abs(mean_rev) > 0,
            abs(momentum) > 0,
            abs(volume) > 0,
            abs(volatility) > 0,
            abs(ml_pred) > 0
        ])
        score = (
            trend_score * self.signal_weights['trend_following'] +
            mean_rev * self.signal_weights['mean_reversion'] +
            momentum * self.signal_weights['momentum'] +
            volume * self.signal_weights['volume_analysis'] +
            volatility * self.signal_weights['volatility'] +
            ml_pred * self.signal_weights['ml_prediction']
        )
        confidence = min(1.0, abs(score))
        # Final signal
        if confidence > self.confidence_threshold and confluence >= self.min_confluence:
            signal = 'BUY' if score > 0 else 'SELL'
        else:
            signal = 'HOLD'
        return signal, confidence, reasons

    def trend_following_signal(self, df, latest, prev, reasons):
        c, s20, s50 = latest['close'], latest['sma_20'], latest['sma_50']
        if pd.isna(c) or pd.isna(s20) or pd.isna(s50): return 0
        if c > s20 > s50:
            reasons.append("Uptrend")
            return 1
        elif c < s20 < s50:
            reasons.append("Downtrend")
            return -1
        return 0

    def mean_reversion_signal(self, df, latest, prev, reasons):
        c, bb_upper, bb_lower = latest['close'], latest['bb_upper'], latest['bb_lower']
        if pd.isna(c) or pd.isna(bb_upper) or pd.isna(bb_lower): return 0
        if c > bb_upper:
            reasons.append("Overbought (BB)")
            return -1
        elif c < bb_lower:
            reasons.append("Oversold (BB)")
            return 1
        return 0

    def momentum_signal(self, df, latest, prev, reasons):
        macd, macd_signal = latest['macd'], latest['macd_signal']
        if pd.isna(macd) or pd.isna(macd_signal): return 0
        if macd > macd_signal:
            reasons.append("Positive momentum (MACD)")
            return 1
        elif macd < macd_signal:
            reasons.append("Negative momentum (MACD)")
            return -1
        return 0

    def volume_analysis_signal(self, df, latest, prev, reasons):
        obv, ad_line = latest['obv'], latest['ad_line']
        if pd.isna(obv) or pd.isna(ad_line): return 0
        if obv > prev['obv'] and ad_line > prev['ad_line']:
            reasons.append("Volume confirms up move")
            return 1
        elif obv < prev['obv'] and ad_line < prev['ad_line']:
            reasons.append("Volume confirms down move")
            return -1
        return 0

    def volatility_signal(self, df, latest, prev, reasons):
        vol = latest['volatility']
        if pd.isna(vol): return 0
        if vol > df['volatility'].mean():
            reasons.append("High volatility")
            return 1
        else:
            return 0

    def ml_prediction_signal(self, mlp, reasons):
        # mlp = (prob_down, prob_up)
        if mlp and len(mlp) == 2:
            if mlp[1] > 0.6:
                reasons.append("ML predicts up")
                return 1
            elif mlp[0] > 0.6:
                reasons.append("ML predicts down")
                return -1
        return 0