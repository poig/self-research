from qbraid.runtime import QbraidProvider

# Use the key provided by the user in the previous step
provider = QbraidProvider()

print("Fetching devices...")
devices = provider.get_devices()

print(f"Found {len(devices)} devices.")
for d in devices:
    print(f"ID: {d.id}, Status: {d.status}")
