from qbraid.runtime import QbraidProvider

provider = QbraidProvider()
devices = provider.get_devices()

print(f"{'ID':<25} | {'Status':<15} | {'Queue':<5} | {'Sim?':<5} | {'Online?':<5}")
print("-" * 65)

for d in devices:
    dev_id = d.id
    
    # Check Status
    try:
        status_val = d.status() if callable(d.status) else d.status
        status_str = str(status_val).upper()
    except Exception as e:
        status_str = f"ERR: {e}"
        
    # Check Queue
    try:
        queue = d.queue_depth() if callable(getattr(d, 'queue_depth', None)) else getattr(d, 'queue_depth', 'N/A')
    except Exception:
        queue = "?"
        
    # Check Simulator
    is_sim = ('sim' in dev_id.lower() or 
              'sv1' in dev_id.lower() or 
              'tn1' in dev_id.lower() or 
              'dm1' in dev_id.lower())
              
    # Check Online
    is_online = (status_str == 'ONLINE')
    
    print(f"{dev_id:<25} | {status_str:<15} | {queue:<5} | {is_sim!s:<5} | {is_online!s:<5}")
