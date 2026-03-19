from app import run_pipeline

print("Running pipeline manually with ensemble...")
try:
    success = run_pipeline(symbol="BTCUSDT", interval="4h", n_states=5, limit=500, use_ensemble=True, force=True)
    print("Success:", success)
except Exception as e:
    import traceback
    traceback.print_exc()
