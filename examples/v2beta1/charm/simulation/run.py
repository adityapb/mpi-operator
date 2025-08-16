from simulate import Simulation, Job, plot_utilization, get_stats
from random import randint, seed, choice
import numpy as np
import sys
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee'])
#plt.rcParams.update({'font.size': 18})

seed(42)

class StencilJob(Job):
    def __init__(self, job_name, min_replicas, max_replicas, priority, **kwargs):
        super().__init__(job_name, min_replicas, max_replicas, priority)
        self.n = kwargs.pop('n', 16)
        self.max_pes = kwargs.pop('max_pes', 60)
        self.niters = kwargs.pop('niters', 10000)

        self.data = {
            512     : [(2, 0.003), (4, 0.0017), (8, 0.000996)],
            2048    : [(4, 0.012), (8, 0.0067), (16, 0.0037)],
            8192    : [(8, 0.049), (16, 0.03), (32, 0.019)],
            16384   : [(16, 0.13), (32, 0.075), (60, 0.046)]
        }

        self.models = {}
        self.model_replicas = {}
        for n, ndata in self.data.items():
            self.models[n] = [None, None]
            self.model_replicas[n] = [ndata[1][0], ndata[2][0]]
            for i in range(len(ndata)-1):
                x = [ndata[i][0], ndata[i+1][0]]
                y = [ndata[i][1], ndata[i+1][1]]
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                self.models[n][i] = (m, c, (x, y))

        self.lbdata = {
            512     : [(2, 0.006), (4, 0.006), (8, 0.006)],
            2048    : [(4, 0.01563), (8, 0.01563), (16, 0.01563)],
            8192    : [(8, 0.181229), (16, 0.181229), (32, 0.181229)],
            16384   : [(16, 0.728237), (32, 0.728237), (60, 0.728237)]
        }

        self.lbmodels = {}
        self.lbmodel_replicas = {}
        for n, ndata in self.lbdata.items():
            self.lbmodels[n] = [None, None]
            self.lbmodel_replicas[n] = [ndata[1][0], ndata[2][0]]
            for i in range(len(ndata)-1):
                x = [ndata[i][0], ndata[i+1][0]]
                y = [ndata[i][1], ndata[i+1][1]]
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                self.lbmodels[n][i] = (m, c)

        self.connect_time_pe = [2, 4, 8, 16, 32, 60]
        self.connect_time_t = [0.634, 0.851, 1.195, 2.55, 4.17, 9.02]
        # Fit linear regression model for connect time
        A = np.vstack([self.connect_time_pe, np.ones(len(self.connect_time_pe))]).T
        self.connect_time_model = np.linalg.lstsq(A, self.connect_time_t, rcond=None)[0]

    def get_connect_time(self):
        return self.connect_time_model[0] * (self.replicas + 1) + self.connect_time_model[1]

    def get_runtime(self):
        models = self.models[self.n]
        replicas = self.model_replicas[self.n]

        for i, r in enumerate(replicas):
            if (self.replicas + 1) <= r:
                m, c, (x, y) = models[i]
                break

        return m * (self.replicas + 1) + c

    def get_completion_time(self):
        return self.get_runtime() * self.niters * (1 - self.completion_fraction)
    
    def get_rescale_overhead(self):
        models = self.lbmodels[self.n]
        replicas = self.lbmodel_replicas[self.n]

        for i, r in enumerate(replicas):
            if (self.replicas + 1) <= r:
                m, c = models[i]

        lbtime = m * (self.replicas + 1) + c

        #print("LBTIME:", self.replicas, lbtime)
    
        return lbtime + self.get_connect_time()
    
    def get_startup_overhead(self):
        return self.get_connect_time()
    
    def update_runtime(self, runtime):
        fraction = runtime / (self.get_runtime() * self.niters * (1 - self.completion_fraction))
        #print(self.completion_fraction, fraction)
        self.completion_fraction += fraction * (1 - self.completion_fraction)
    

def generate_job_list(nexps):
    jobs = []
    indices = [0, 1, 2, 3]
    sizes_per_pe = [256, 512, 1024, 1024]
    min_pes = [2, 4, 8, 16]
    timesteps_per_job = [40000, 40000, 40000, 10000]
    job_prefixes = ["small", "medium", "large", "xlarge"]
    njobs = 16

    for n in range(nexps):
        counts = [0, 0, 0, 0]
        jobs_list = []
        for i in range(njobs):
            idx = choice(indices)
            #priority = (3 - idx) + randint(1, 2)
            priority = randint(1, 5)
            min_replicas = min_pes[idx] - 1
            max_replicas = min(4 * (min_replicas + 1), 60) - 1
            problem_size = (min_replicas + 1) * sizes_per_pe[idx]
            timesteps = timesteps_per_job[idx] #+ 100 * randint(0, 10)
            prefix = job_prefixes[idx]
            #create_job(prefix, i, priority, problem_size, min_replicas, max_replicas, timesteps)

            jobs_list.append(StencilJob("charm-%s-%i" % (prefix, counts[idx]), min_replicas, max_replicas, 
                                        priority, n=problem_size, niters=timesteps))
            counts[idx] += 1
        jobs.append(jobs_list)
    return jobs


def run_simulation(jobs_list, mode, max_pes, job_submission_time, rescale_gap):
    if mode == "min_replicas":
        for job in jobs_list:
            job.max_replicas = job.min_replicas
    elif mode == "max_replicas":
        for job in jobs_list:
            job.min_replicas = job.max_replicas    
    
    if mode == "elastic":
        simulator = Simulation(max_pes, rescale_gap)
    else:
        simulator = Simulation(max_pes, 100000 * 60)
    
    events = simulator.simulate([job_submission_time*i for i in range(len(jobs_list))], jobs_list)
    return get_stats(events, max_pes)

def vary_submission_time(modes, max_pes):
    nexperiments = 100
    jobs = generate_job_list(nexperiments)

    total, response, completion, util = {}, {}, {}, {}
    for m in modes:
        total[m] = []
        response[m] = []
        completion[m] = []
        util[m] = []

    submission_times = [30*i for i in range(13)]
    for t in submission_times:
        for m in modes:
            print(f"Running simulation for mode {m} with submission time {t}")
            final_times, mean_responses, mean_completions, utilizations = [], [], [], []
            for i in range(nexperiments):
                final_time, mean_response, mean_completion, utilization = run_simulation(deepcopy(jobs[i]), m, max_pes, t, 3*60)
                final_times.append(final_time)
                mean_responses.append(mean_response)
                mean_completions.append(mean_completion)
                utilizations.append(utilization)

            avg_final_time = np.mean(final_times)
            avg_mean_response = np.mean(mean_responses)
            avg_mean_completion = np.mean(mean_completions)
            avg_utilization = np.mean(utilizations)

            total[m].append(avg_final_time)
            response[m].append(avg_mean_response)
            completion[m].append(avg_mean_completion)
            util[m].append(avg_utilization)

    data = {
        'submission_time': submission_times
    }

    for m in modes:
        data[f'total_{m}'] = total[m]
        data[f'response_{m}'] = response[m]
        data[f'completion{m}'] = completion[m]
        data[f'util_{m}'] = util[m]

    df = pd.DataFrame(data)
    df.to_csv('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/results.csv', index=False)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_']

    for i, m in enumerate(modes):
        plt.plot(submission_times, total[m], label=m)
    plt.xlabel('Submission Gap (s)')
    plt.ylabel('Total Time (s)')
    #plt.title('Total Time vs Submission Gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/total_time_plot.pdf')
    plt.close()

    for i, m in enumerate(modes):
        plt.plot(submission_times, response[m], label=m)
    plt.xlabel('Submission Gap (s)')
    plt.ylabel('Response Time (s)')
    #plt.title('Response Time vs Submission Gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/response_time_plot.pdf')
    plt.close()

    for i, m in enumerate(modes):
        plt.plot(submission_times, util[m], label=m)
    plt.xlabel('Submission Gap (s)')
    plt.ylabel('Utilization')
    #plt.title('Utilization vs Submission Gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/utilization_plot.pdf')
    plt.close()

    for i, m in enumerate(modes):
        plt.plot(submission_times, completion[m], label=m)
    plt.xlabel('Submission Gap (s)')
    plt.ylabel('Completion Time (s)')
    #plt.title('Completion Time vs Submission Gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/completion_time_plot.pdf')
    plt.close()

    print(df)

def vary_rescale_gap(modes, max_pes):
    nexperiments = 100
    jobs = generate_job_list(nexperiments)

    total, response, completion, util = {}, {}, {}, {}
    for m in modes:
        total[m] = []
        response[m] = []
        completion[m] = []
        util[m] = []

    rescale_gaps = [120*i for i in range(11)]
    for t in rescale_gaps:
        for m in modes:
            if m != "elastic" and len(total[m]) > 0:
                total[m].append(total[m][0])
                response[m].append(response[m][0])
                completion[m].append(completion[m][0])
                util[m].append(util[m][0])
                continue
            print(f"Running simulation for mode {m} with rescale gap {t}")
            final_times, mean_responses, mean_completions, utilizations = [], [], [], []
            for i in range(nexperiments):
                final_time, mean_response, mean_completion, utilization = run_simulation(deepcopy(jobs[i]), m, max_pes, 180, t)
                final_times.append(final_time)
                mean_responses.append(mean_response)
                mean_completions.append(mean_completion)
                utilizations.append(utilization)

            avg_final_time = np.mean(final_times)
            avg_mean_response = np.mean(mean_responses)
            avg_mean_completion = np.mean(mean_completions)
            avg_utilization = np.mean(utilizations)

            total[m].append(avg_final_time)
            response[m].append(avg_mean_response)
            completion[m].append(avg_mean_completion)
            util[m].append(avg_utilization)

    data = {
        'rescale_gap': rescale_gaps
    }

    for m in modes:
        data[f'total_{m}'] = total[m]
        data[f'response_{m}'] = response[m]
        data[f'completion{m}'] = completion[m]
        data[f'util_{m}'] = util[m]

    df = pd.DataFrame(data)
    df.to_csv('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/results_rescale.csv', index=False)
    markers = ['x', '', '', '', 'v', '<', '>', 'p', '*', 'h', 'H', 'o', 'd', '|', '_']

    #plt.figure(figsize=(12, 8))

    for i, m in enumerate(modes):
        plt.plot(rescale_gaps, total[m], label=m, marker=markers[i % len(markers)])
    plt.xlabel('Rescale Gap (s)')
    plt.ylabel('Total Time (s)')
    #plt.title('Total Time vs Rescale gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/total_time_plot_rescale.pdf')
    #plt.show()
    plt.close()

    for i, m in enumerate(modes):
        plt.plot(rescale_gaps, response[m], label=m, marker=markers[i % len(markers)])
    plt.xlabel('Rescale Gap (s)')
    plt.ylabel('Response Time (s)')
    #plt.title('Response Time vs Rescale gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/response_time_plot_rescale.pdf')
    #plt.show()
    plt.close()

    #plt.figure(figsize=(12, 8))

    for i, m in enumerate(modes):
        plt.plot(rescale_gaps, util[m], label=m, marker=markers[i % len(markers)])
    plt.xlabel('Rescale Gap (s)')
    plt.ylabel('Utilization')
    #plt.title('Utilization vs Rescale gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/utilization_plot_rescale.pdf')
    #plt.show()
    plt.close()

    #plt.figure(figsize=(12, 8))

    for i, m in enumerate(modes):
        plt.plot(rescale_gaps, completion[m], label=m, marker=markers[i % len(markers)])
    plt.xlabel('Rescale Gap (s)')
    plt.ylabel('Completion Time (s)')
    #plt.title('Completion Time vs Rescale gap')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/aditya/mpi-operator/examples/v2beta1/charm/simulation/completion_time_plot_rescale.pdf')
    #plt.show()
    plt.close()

    print(df)

def find_critical_experiment(modes, max_pes):
    """
    Finds the experiment where the elastic mode has the largest minimum performance
    improvement over all other specified modes.
    """
    nexperiments = 1000
    jobs = generate_job_list(nexperiments)

    largest_min_improvement = -1
    critical_experiment_index = -1
    critical_experiment_results = {}
    critical_total_times = {}
    critical_job_list = None

    print(f"Searching for critical experiment across {nexperiments} random job sets...")

    for i in range(nexperiments):
        current_jobs = jobs[i]
        completion_times = {}
        total_times = {}
        
        for m in modes:
            final_time, _, mean_completion, _ = run_simulation(deepcopy(current_jobs), m, max_pes, 90, 3*60)
            completion_times[m] = mean_completion
            total_times[m] = final_time

        elastic_completion = completion_times.get("elastic")
        if elastic_completion is None:
            continue

        other_completions = [completion_times[m] for m in modes if m != "elastic"]
        if not other_completions:
            continue
        
        # Calculate improvement of elastic over other modes. Positive value means elastic is faster.
        improvements = [c - elastic_completion for c in other_completions]
        current_min_improvement = min(improvements)

        if current_min_improvement > largest_min_improvement and total_times["elastic"] < 2000 and total_times["elastic"] < min([total_times[m] for m in modes if m != "elastic"]):
            largest_min_improvement = current_min_improvement
            critical_experiment_index = i
            critical_experiment_results = completion_times
            critical_total_times = total_times
            critical_job_list = deepcopy(current_jobs)
            print(f"New critical experiment found at index {i} with largest min improvement {largest_min_improvement:.2f}s")

    print("\n--- Critical Experiment Analysis ---")
    if critical_experiment_index != -1:
        print(f"The experiment with the largest minimum performance improvement was found at index: {critical_experiment_index}")
        print(f"Largest minimum improvement observed: {largest_min_improvement:.2f}s")
        print("Mean completion times for this experiment:")
        for mode, time in critical_experiment_results.items():
            print(f"  - {mode}: {time:.2f}s")

        print("\nTotal times for this experiment:")
        for mode, time in critical_total_times.items():
            print(f"  - {mode}: {time:.2f}s")
        
        print("\nJob details for this experiment:")
        for job in critical_job_list:
            print(f"  - {job.job_name}: priority={job.priority}, min={job.min_replicas}, max={job.max_replicas}, n={job.n}, niters={job.niters}")
    else:
        print("Could not find a critical experiment where elastic was consistently faster.")

    return critical_job_list

if __name__ == '__main__':
    sizes = [256, 512, 1024, 1024]
    sizes_per_pe = [256, 512, 1024, 1024]
    min_pes = [2, 4, 8, 16]
    timesteps_per_job = [40000, 40000, 40000, 10000]
    job_prefixes = ["small", "medium", "large", "xlarge"]
    counts = [0, 0, 0, 0]
    njobs = 16
    #njobs = 9

    #jobs = [2, 3, 2, 1, 0, 1, 1, 1, 0, 1, 1, 2, 0, 1, 1, 1]
    #priorities = [2, 1, 3, 3, 4, 3, 3, 3, 5, 3, 3, 2, 4, 3, 3, 3]
    jobs = [2, 1, 1, 0, 3, 3, 0, 3, 1, 1, 0, 3, 0, 1, 1, 1]
    priorities = [2, 3, 4, 4, 2, 1, 4, 1, 4, 3, 4, 1, 5, 3, 3, 3]
    #jobs = [2, 3, 2, 1, 0, 1, 1, 1, 0]

    jobs_list = []
    for i, job_index in enumerate(jobs):
        idx = job_index
        #priority = (3 - job_index) + randint(1, 2)
        #print(priority)
        priority = priorities[i]
        min_replicas = min_pes[idx] - 1
        max_replicas = min(4 * (1 + min_replicas), 60) - 1
        problem_size = (min_replicas + 1) * sizes_per_pe[idx]
        timesteps = timesteps_per_job[idx] #+ 100 * randint(0, 10)
        prefix = job_prefixes[idx]
        #create_job(prefix, i, priority, problem_size, min_replicas, max_replicas, timesteps)

        jobs_list.append(StencilJob("charm-%s-%i" % (prefix, counts[idx]), min_replicas, max_replicas, 
                                    priority, n=problem_size, niters=timesteps))
        counts[idx] += 1

    print(run_simulation(deepcopy(jobs_list), "elastic", 60, 90, 3*60))
    print(run_simulation(deepcopy(jobs_list), "moldable", 60, 90, 3*60))
    print(run_simulation(deepcopy(jobs_list), "min_replicas", 60, 90, 3*60))
    print(run_simulation(deepcopy(jobs_list), "max_replicas", 60, 90, 3*60))
    #simulator = Simulation(60, 3 * 60000)
    #events = simulator.simulate([90*i for i in range(njobs)], jobs_list)
    #jobs = ["charm-small-%i" % i for i in range(16)] + ["charm-medium-%i" % i for i in range(16)] + \
    #    ["charm-large-%i" % i for i in range(16)] + ["charm-xlarge-%i" % i for i in range(16)]
    #plot_utilization(events, jobs, 60)
    #print(events)
    #print(get_stats(events, 60))
    #vary_rescale_gap(["elastic", "moldable", "min_replicas", "max_replicas"], 60)
    #vary_submission_time(["elastic", "moldable", "min_replicas", "max_replicas"], 60)
    #find_critical_experiment(["elastic", "moldable", "min_replicas", "max_replicas"], 60)