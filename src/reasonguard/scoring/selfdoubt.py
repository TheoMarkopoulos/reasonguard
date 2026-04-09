def compute_selfdoubt_score(
    hvr: float,
    verbalized_conf: float,
    mu_hvr: float,
    sigma_hvr: float,
    mu_verb: float,
    sigma_verb: float,
) -> float:
    """Compute full SELFDOUBT score via z-score fusion.

    score = z(-HVR) + z(Verb)

    Higher score = more reliable/confident.
    """
    z_hvr = (-hvr - mu_hvr) / (sigma_hvr + 1e-8)
    z_verb = (verbalized_conf - mu_verb) / (sigma_verb + 1e-8)
    return z_hvr + z_verb
