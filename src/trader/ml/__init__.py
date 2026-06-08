"""ML model integrations."""

from trader.ml.runtime_compat import install_observational_score_alias

install_observational_score_alias()

__all__ = ["install_observational_score_alias"]
