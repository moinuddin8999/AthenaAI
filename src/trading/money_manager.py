from datetime import datetime, timezone
import logging
from src.config import BotConfig

log = logging.getLogger("AIBot")

class MoneyManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.daily_pnl = 0.0
        self.day_start = datetime.now(timezone.utc).date()

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day_start:
            log.info("New day — resetting daily P&L tracker")
            self.daily_pnl = 0.0
            self.day_start = today

    def can_trade(self) -> bool:
        self.reset_if_new_day()
        return self.daily_pnl > -self.cfg.max_daily_loss

    def compute_stake(self, confidence: float, win_rate: float, payout: float = 0.85) -> float:
        """Kelly criterion capped by config limits, confidence-weighted."""
        if payout <= 0:
            return self.cfg.base_stake
        # Blend historical win_rate with model confidence for the probability estimate
        prob = 0.7 * confidence + 0.3 * win_rate
        # Kelly: f* = (p*b - q) / b  where b=payout, p=prob, q=1-p
        edge = prob * payout - (1 - prob)
        if edge <= 0:
            return self.cfg.base_stake
        kelly = edge / payout
        fraction = kelly * self.cfg.kelly_fraction
        stake = self.cfg.base_stake + fraction * (self.cfg.max_stake - self.cfg.base_stake)
        stake = max(self.cfg.base_stake, min(stake, self.cfg.max_stake))
        return round(stake, 2)

    def record(self, pnl: float):
        self.daily_pnl += pnl
