from qbraid.runtime import QbraidProvider
from qiskit import QuantumCircuit

provider = QbraidProvider()
devices = provider.get_devices()

# Create a dummy circuit for estimation
qc = QuantumCircuit(1)
qc.h(0)
qc.measure_all()

print(f"{'ID':<25} | {'Status':<15} | {'Cost Estimate':<15}")
print("-" * 60)

for d in devices:
    dev_id = d.id
    
    # Filter for the candidates we found earlier
    if dev_id not in ['rigetti_ankaa_3', 'iqm_emerald', 'ionq_simulator']:
        continue
        
    try:
        # Try the preflight method suggested by user
        # Note: This might be provider-specific (e.g. IonQ)
        # We wrap in try/except to handle devices that don't support it
        job = d.run(qc, shots=100, preflight=True)
        
        # Wait for result (preflight should be fast)
        job.wait_for_final_state()
        
        # Check metadata
        meta = job.metadata()
        cost = meta.get("cost_usd", "N/A")
        if cost == "N/A":
             cost = meta.get("cost", "N/A")
             
        print(f"{dev_id:<25} | {d.status():<15} | ${cost}")
        
    except Exception as e:
        print(f"{dev_id:<25} | {d.status():<15} | Error: {str(e)[:30]}...")
