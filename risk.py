def calculate_position_size(cash, confidence, volatility):
    # Volatility and confidence scaled position sizing. Max 5% of cash.
    base_risk_pct = 0.01  # 1% base risk
    conf_factor = min(2.0, max(0.5, confidence * 2))
    vol_factor = 1 / max(0.01, volatility)
    risk = base_risk_pct * conf_factor * vol_factor
    return min(0.05, risk)  # Cap to 5% of cash

def get_risk_management_levels(entry_price, stop_loss_pct=0.015, take_profit_pct=0.03):
    stop_loss = entry_price * (1 - stop_loss_pct)
    take_profit = entry_price * (1 + take_profit_pct)
    return {'stop_loss': stop_loss, 'take_profit': take_profit}