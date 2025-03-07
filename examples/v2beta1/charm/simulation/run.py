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
            512     : [(2, 0.000996), (4, 0.000675), (8, 0.000627)],
            2048    : [(4, 0.00328), (8, 0.0021), (16, 0.0023)],
            8192    : [(8, 0.0325), (16, 0.0275), (32, 0.016)],
            16384   : [(16, 0.11), (32, 0.064), (59, 0.035)]
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
                self.models[n][i] = (m, c)

        self.lbdata = {
            512     : [(2, 0.006), (4, 0.006), (8, 0.006)],
            2048    : [(4, 0.0097), (8, 0.0097), (16, 0.046659)],
            8192    : [(8, 0.61581), (16, 2.934641), (32, 25.083405)],
            16384   : [(16, 14.601492), (32, 95.771426), (59, 59.793259)]
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

        self.connect_time_pe = [2, 4, 8, 16, 32, 59]
        self.connect_time_t = [1.248, 1.492, 3.902, 5.035, 6.949, 12.25]
        # Fit linear regression model for connect time
        A = np.vstack([self.connect_time_pe, np.ones(len(self.connect_time_pe))]).T
        self.connect_time_model = np.linalg.lstsq(A, self.connect_time_t, rcond=None)[0]

    def get_connect_time(self):
        return self.connect_time_model[0] * self.replicas + self.connect_time_model[1]

    def get_runtime(self):
        models = self.models[self.n]
        replicas = self.model_replicas[self.n]

        for i, r in enumerate(replicas):
            if self.replicas <= r:
                m, c = models[i]

        return m * self.replicas + c

    def get_completion_time(self):
        return self.get_runtime() * self.niters * (1 - self.completion_fraction)
    
    def get_rescale_overhead(self):
        models = self.lbmodels[self.n]
        replicas = self.lbmodel_replicas[self.n]

        for i, r in enumerate(replicas):
            if self.replicas <= r:
                m, c = models[i]

        lbtime = m * self.replicas + c

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
            priority = (3 - idx) + randint(1, 2)
            #priority = randint(1, 5)
            min_replicas = min_pes[idx]
            max_replicas = min(4 * min_replicas, 59)
            problem_size = min_replicas * sizes_per_pe[idx]
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
        simulator = Simulation(60, rescale_gap)
    else:
        simulator = Simulation(60, 100000 * 60)
    
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
                final_time, mean_response, mean_completion, utilization = run_simulation(deepcopy(jobs[i]), m, max_pes, t, 5*60)
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

if __name__ == '__main__':
    sizes = [256, 512, 1024, 1024]
    sizes_per_pe = [256, 512, 1024, 1024]
    min_pes = [2, 4, 8, 16]
    timesteps_per_job = [40000, 40000, 40000, 10000]
    job_prefixes = ["small", "medium", "large", "xlarge"]
    counts = [0, 0, 0, 0]
    njobs = 16
    #njobs = 9

    jobs = [2, 3, 2, 1, 0, 1, 1, 1, 0, 1, 1, 2, 0, 1, 1, 1]
    #jobs = [2, 3, 2, 1, 0, 1, 1, 1, 0]

    jobs_list = []
    for i, job_index in enumerate(jobs):
        idx = job_index
        priority = (3 - job_index) + randint(1, 2)
        min_replicas = min_pes[idx]
        max_replicas = min(4 * min_replicas, 59)
        problem_size = min_replicas * sizes_per_pe[idx]
        timesteps = timesteps_per_job[idx] #+ 100 * randint(0, 10)
        prefix = job_prefixes[idx]
        #create_job(prefix, i, priority, problem_size, min_replicas, max_replicas, timesteps)

        jobs_list.append(StencilJob("charm-%s-%i" % (prefix, counts[idx]), max_replicas, max_replicas, 
                                    priority, n=problem_size, niters=timesteps))
        counts[idx] += 1
        
    simulator = Simulation(60, 3 * 60)
    events = simulator.simulate([90*i for i in range(njobs)], jobs_list)
    jobs = ["charm-small-%i" % i for i in range(16)] + ["charm-medium-%i" % i for i in range(16)] + \
        ["charm-large-%i" % i for i in range(16)] + ["charm-xlarge-%i" % i for i in range(16)]
    #plot_utilization(events, jobs, 60)
    #print(events)
    print(get_stats(events, 60))
    #vary_rescale_gap(["elastic", "moldable", "min_replicas", "max_replicas"], 60)
    vary_submission_time(["elastic", "moldable", "min_replicas", "max_replicas"], 60)