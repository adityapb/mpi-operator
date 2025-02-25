import subprocess
import time

output_file = "/home/aditya/mpi-operator/examples/v2beta1/charm/pod_utilization_long2.log"

open(output_file, "w").close()  # Clear the file before starting

while True:
    result = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True)
    with open(output_file, "a") as file:
        file.write(result.stdout)
        file.write("\n")
        file.write("LOG> %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        file.write("\n")
    time.sleep(2)