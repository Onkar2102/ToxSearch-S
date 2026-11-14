# RQ1 mechanism diagnostics

## Cross-source edge share at t=0.10

- `d_E`: cross-source=0.099, hub1% edges=0.162, maxdeg=159, giant=0.106
- `d_negE`: cross-source=0.455, hub1% edges=0.488, maxdeg=421, giant=0.615

## Hub removal (top 1% degree)

- d_E@0.10: giant 0.106 → 0.107
- d_negE@0.10: giant 0.615 → 0.487
- d_negE@dens-match(0.33): giant 0.767 → 0.670
- d_E@dens-match(0.07): giant 0.060 → 0.053

## Feasible operating windows (D_asg<0.70 & assigned≥70% & n_mat≥3)

- `d_E` incdbscan MinPts=2.0: t∈[0.10, 0.15] (2 points)
- `d_E` incdbscan MinPts=3.0: t∈[0.10, 0.15] (2 points)
- `d_E` incdbscan MinPts=5.0: t∈[0.15, 0.15] (1 points)
- `d_E` leader_follower MinPts=: t∈[0.15, 0.50] (8 points)
- `d_EP` incdbscan MinPts=2.0: t∈[0.10, 0.10] (1 points)
- `d_EP` incdbscan MinPts=3.0: t∈[0.10, 0.10] (1 points)
- `d_EP` incdbscan MinPts=5.0: t∈[0.10, 0.10] (1 points)
- `d_EP` leader_follower MinPts=: t∈[0.10, 0.40] (7 points)
- `d_N` leader_follower MinPts=: t∈[0.30, 0.50] (5 points)
