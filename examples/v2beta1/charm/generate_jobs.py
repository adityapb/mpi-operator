from random import randint, seed, choice, shuffle
import sys
import os
import time

seed(42)

size_index = [0, 1, 2]
sizes_per_pe = [1024, 1024, 2048]
min_pes = [1, 4, 4]
timesteps_per_job = [1000, 2000, 500]
job_prefixes = ["small", "medium", "large"]

njobs = 20


def create_job(prefix, job_index, priority, problem_size, min_replicas, max_replicas, timesteps):
    with open("charm-template.yaml", "r") as file:
        template = file.read()
    
    odf = 2
    num_chares = odf * max_replicas
    num_chares = 2 ** (num_chares - 1).bit_length()
    chare_size = problem_size // num_chares

    job_yaml = template.format(
        prefix=prefix,
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
    jobs = [0] * int(njobs * 0.25)
    jobs += [1] * int(njobs * 0.5)
    jobs += [2] * int(njobs * 0.25)
    shuffle(jobs)
    print(jobs)
    for i, job_index in enumerate(jobs):
        idx = job_index
        priority = 3 - idx + randint(0, 3)
        min_replicas = min_pes[idx]
        max_replicas = 4 * min_replicas
        problem_size = min_replicas * sizes_per_pe[idx]
        timesteps = timesteps_per_job[idx] + 100 * randint(0, 10)
        prefix = job_prefixes[idx]
        create_job(prefix, i, priority, problem_size, min_replicas, max_replicas, timesteps)


def submit_jobs():
    for job_index in range(njobs):
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