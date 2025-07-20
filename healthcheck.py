import sys
import urllib.request

# This script is designed to be run by Docker's HEALTHCHECK.
# It attempts to access the addon's manifest.
# - Exits with code 0 (success) if the manifest is reachable and healthy.
# - Exits with code 1 (failure) for any error.

try:
    # Set a timeout to prevent the check from hanging indefinitely.
    response = urllib.request.urlopen("http://localhost:8000/manifest.json", timeout=5)

    # Check for a healthy HTTP status code (2xx or 3xx).
    if response.status >= 400:
        print(f"Health check failed: Received status code {response.status}")
        sys.exit(1)
    
    # If we reached here, the server is responding correctly.
    print("Health check passed.")
    sys.exit(0)

except Exception as e:
    # Catch any exception, like connection refused, timeout, etc.
    print(f"Health check failed with exception: {e}")
    sys.exit(1)
