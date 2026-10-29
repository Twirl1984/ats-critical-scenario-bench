from hypothesis import HealthCheck, settings

# deterministisch: feste Beispielfolge, keine Datenbank, reproduzierbar in CI und lokal
settings.register_profile("det", derandomize=True, database=None, max_examples=150, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("det")
