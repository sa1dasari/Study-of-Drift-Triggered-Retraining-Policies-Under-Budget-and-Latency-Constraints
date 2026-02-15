# Total Experiment Combinations
## Drift Types (3):
1. Abrupt (drift_point=5000, magnitude=0.5)
2. Gradual (smooth transition over window)
3. Recurring (periodic pattern)

## Policies (3):
1. Periodic (interval-based)
2. ErrorThreshold (performance-based)
3. DriftTrigger (detection-based)

## Budgets (3):
1. K=5 (very tight: 5 retrains for 10k samples)
2. K=10 (moderate: 10 retrains)
3. K=20 (loose: 20 retrains)

## Latency Levels (3):
1. Low: retrain_latency=10, deploy_latency=1 (total=11 steps)
2. Medium: retrain_latency=100, deploy_latency=5 (total=105 steps)
3. High: retrain_latency=500, deploy_latency=20 (total=520 steps)

## Random Seeds:
- 3 (for stability/variance measurement)

## Total:
### 3 drift × 3 policies × 3 budgets × 3 latency = 81 configs
### 81 × 3 seeds = 243 runs