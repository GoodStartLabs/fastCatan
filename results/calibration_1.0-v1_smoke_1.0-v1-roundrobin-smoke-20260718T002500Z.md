# 1.0-v1 smoke calibration

Run `1.0-v1-roundrobin-smoke-20260718T002500Z`; 256 games per directed opponent/mode; wall 602.1s; 34607 decisions/s.

## trading_on

| Candidate | random-legal | builder-basic | builder-strong | trade-happy | trade-averse | balanced-strong | catanatron-value | oracle-ab-d2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random-legal | — | 0.035 | 0.020 | 0.000 | 0.031 | 0.020 | 0.000 | 0.000 |
| builder-basic | 0.656 | — | 0.121 | 0.113 | 0.129 | 0.113 | 0.008 | 0.016 |
| builder-strong | 0.863 | 0.512 | — | 0.238 | 0.281 | 0.258 | 0.133 | 0.195 |
| trade-happy | 0.863 | 0.512 | 0.258 | — | 0.281 | 0.258 | 0.133 | 0.195 |
| trade-averse | 0.863 | 0.438 | 0.258 | 0.211 | — | 0.254 | 0.133 | 0.094 |
| balanced-strong | 0.863 | 0.512 | 0.242 | 0.254 | 0.281 | — | 0.133 | 0.195 |
| catanatron-value | 1.000 | 0.871 | 0.543 | 0.508 | 0.570 | 0.539 | — | 0.188 |
| oracle-ab-d2 | 1.000 | 0.922 | 0.520 | 0.523 | 0.668 | 0.520 | 0.297 | — |

| Rank | Agent | Mean win share |
|---:|---|---:|
| 1 | oracle-ab-d2 | 0.6356 |
| 2 | catanatron-value | 0.6027 |
| 3 | trade-happy | 0.3571 |
| 4 | balanced-strong | 0.3544 |
| 5 | builder-strong | 0.3544 |
| 6 | trade-averse | 0.3214 |
| 7 | builder-basic | 0.1652 |
| 8 | random-legal | 0.0151 |

## trading_off

| Candidate | random-legal | builder-basic | builder-strong | trade-happy | trade-averse | balanced-strong | catanatron-value | oracle-ab-d2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random-legal | — | 0.031 | 0.012 | 0.012 | 0.012 | 0.012 | 0.000 | 0.000 |
| builder-basic | 0.680 | — | 0.129 | 0.129 | 0.129 | 0.129 | 0.008 | 0.016 |
| builder-strong | 0.859 | 0.438 | — | 0.246 | 0.246 | 0.246 | 0.133 | 0.094 |
| trade-happy | 0.859 | 0.438 | 0.246 | — | 0.246 | 0.246 | 0.133 | 0.094 |
| trade-averse | 0.859 | 0.438 | 0.246 | 0.246 | — | 0.246 | 0.133 | 0.094 |
| balanced-strong | 0.859 | 0.438 | 0.246 | 0.246 | 0.246 | — | 0.133 | 0.094 |
| catanatron-value | 1.000 | 0.871 | 0.570 | 0.570 | 0.570 | 0.570 | — | 0.188 |
| oracle-ab-d2 | 1.000 | 0.922 | 0.668 | 0.668 | 0.668 | 0.668 | 0.297 | — |

| Rank | Agent | Mean win share |
|---:|---|---:|
| 1 | oracle-ab-d2 | 0.6987 |
| 2 | catanatron-value | 0.6200 |
| 3 | balanced-strong | 0.3231 |
| 4 | builder-strong | 0.3231 |
| 5 | trade-averse | 0.3231 |
| 6 | trade-happy | 0.3231 |
| 7 | builder-basic | 0.1741 |
| 8 | random-legal | 0.0112 |

## Sanity verdict

No configured tier inversion detected.
