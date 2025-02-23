from random import randint, seed
import sys
import os
import time

seed(42)

def create_job(job_index, priority, problem_size, min_replicas, max_replicas, timesteps):
    with open("charm-template.yaml", "r") as file:
        template = file.read()
    
    odf = 2
    num_chares = odf * max_replicas
    num_chares = 2 ** (num_chares - 1).bit_length()
    chare_size = problem_size // num_chares

    job_yaml = template.format(
        job_index=job_index,
        priority=priority,
        problem_size=problem_size,
        chare_size=chare_size,
        timesteps=timesteps,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    with open(f"jobs/charm-job-{job_index}.yaml", "w") as file:
        file.write(job_yaml)


def generate_jobs():
    for job_index in range(10):
        priority = randint(1, 5)
        problem_size = 2 ** (randint(9, 12))
        min_replicas = randint(1, 3)
        max_replicas = randint(min_replicas+1, 8)
        timesteps = 100 * randint(20, 30)
        create_job(job_index, priority, problem_size, min_replicas, max_replicas, timesteps)


def submit_jobs():
    for job_index in range(10):
        job_file = f"jobs/charm-job-{job_index}.yaml"
        print(f"Submitting {job_file}")
        # Here you would submit the job using your cluster's job submission command
        os.system(f"kubectl apply -f {job_file}")
        # For this example, we'll just print the command
        #print(f"kubectl apply -f {job_file}")
        time.sleep(10)


if __name__ == "__main__":
    if sys.argv[1] == "generate_jobs":
        generate_jobs()
    elif sys.argv[1] == "submit":
        submit_jobs()
    else:
        print("Invalid argument. Use 'generate_jobs' to generate job files or 'submit' to submit them.")
        sys.exit(1)