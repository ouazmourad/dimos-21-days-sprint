# Heartbeat

Every 30 minutes:
1. Run `scripts/dimos_bridge.py status` to check DimOS connectivity
2. If status changed since last check, log it
3. No need to message user unless something went wrong
