import subprocess
import time
import sys

mode = sys.argv[1]

output_file = f"pod_utilization_{mode}.log"

open(output_file, "w").close()  # Clear the file before starting

while True:
    result = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True)
    with open(output_file, "a") as file:
        file.write(result.stdout)
        file.write("\n")
        file.write("LOG> %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        file.write("\n")
    time.sleep(2)