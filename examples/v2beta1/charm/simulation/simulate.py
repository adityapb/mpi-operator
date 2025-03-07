import heapq
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

EPS = 0.01
DELAY = 1

class Event(object):
    def __init__(self, timestamp, job):
        self.timestamp = timestamp
        self.job = job

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __eq__(self, other):
        if self.job is None and other.job is None:
            return self.timestamp == other.timestamp
        elif not self.job is None and not other.job is None:
            return self.timestamp == other.timestamp and self.job == other.job
        else:
            return False

    def __hash__(self):
        return hash((self.timestamp, self.job, self.event_type))


class SubmitEvent(Event):
    def __init__(self, timestamp, job):
        super().__init__(timestamp, job)

    def __str__(self):
        return "Submit: %s, priority = %i" % (self.job.job_name, self.job.priority)

class StartEvent(Event):
    def __init__(self, timestamp, job, replicas):
        super().__init__(timestamp, job)
        self.replicas = replicas

    def __str__(self):
        return "Start: %s: %i" % (self.job.job_name, self.replicas)

class CompleteEvent(Event):
    def __init__(self, timestamp, job):
        super().__init__(timestamp, job)

    def __str__(self):
        return "Complete: %s" % self.job.job_name

class RescaleEvent(Event):
    def __init__(self, timestamp, job, old_replicas, new_replicas):
        super().__init__(timestamp, job)
        self.old_replicas = old_replicas
        self.new_replicas = new_replicas

    def __str__(self):
        return "Rescale: %s: %i -> %i" % (self.job.job_name, self.old_replicas, self.new_replicas)

class CheckQueueEvent(Event):
    def __init__(self, timestamp, job):
        super().__init__(timestamp, job)

    def __str__(self):
        return "CheckQueue"

class AssignFreeSlotsEvent(Event):
    def __init__(self, timestamp, job):
        super().__init__(timestamp, job)

    def __str__(self):
        return "AssignFreeSlots"

    
class JobList(object):
    def __init__(self):
        self.jobs = []

    def push(self, job):
        self.jobs.append(job)
        sorted(self.jobs, key=lambda x: x)

    def pop(self, index):
        return self.jobs.pop(index)

    def remove(self, x):
        self.jobs.remove(x)

    def __len__(self):
        return len(self.jobs)


class EventHeap(object):
    def __init__(self):
        self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0] if self.heap else None
    
    def remove(self, job):
        for iter in self.heap:
            if isinstance(iter.job, Job) and iter.job.job_name == job.job_name:
                self.heap.remove(iter)
                heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)


class Job(object):
    def __init__(self, job_name, min_replicas, max_replicas, priority):
        self.job_name = job_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.priority = priority
        self.replicas = 0
        self.runtime = 0
        self.overhead = 0
        self.completion_fraction = 0
        self.last_action = 0
        self.submit_time = 0
        self.status = ""

    def set_last_action(self, last_action):
        #print("Setting last action of %s to %f" % (self.job_name, last_action))
        self.last_action = last_action

    def get_completion_time(self):
        raise NotImplementedError("get_completion_time() must be implemented")

    def get_rescale_overhead(self):
        raise NotImplementedError("get_rescale_overhead() must be implemented")
    
    def update_runtime(self, runtime):
        raise NotImplementedError("update_rutime() must be implemented")
    
    def get_startup_overhead(self):
        raise NotImplementedError("IMPlmente")

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.submit_time > other.submit_time
        return self.priority < other.priority
    
    def __eq__(self, other):
        return self.job_name == other.job_name and \
            self.min_replicas == other.min_replicas and \
                self.max_replicas == other.max_replicas and \
                    self.priority == other.priority
    
    def __hash__(self):
        return hash((self.job_name, self.min_replicas, self.max_replicas, self.priority))


class Simulation(object):
    def __init__(self, free_slots, T_rescale_gap):
        self.T_rescale_gap = T_rescale_gap
        self.free_slots = free_slots
        self.event_heap = EventHeap()
        self.running_jobs = JobList()
        self.queued_jobs = JobList()

    def assign_free_workers(self, timestamp):
        running_index = len(self.running_jobs) - 1
        queued_index = len(self.queued_jobs) - 1

        min_time = 1 << 30

        while self.free_slots > 0 and (running_index >= 0 or queued_index >= 0):
            #print(self.free_slots, running_index, queued_index, len(self.queued_jobs))
            launcher_count = 0

            if len(self.running_jobs) > 0 and running_index >= 0:
                running_job = self.running_jobs.jobs[running_index]
            else:
                running_job = None
            
            if len(self.queued_jobs) > 0 and queued_index >= 0:
                queued_job = self.queued_jobs.jobs[queued_index]
            else:
                queued_job = None

            if queued_job is None:
                selected_job = running_job
                running_index -= 1
            elif running_job is None:
                selected_job = queued_job
                self.queued_jobs.pop(queued_index)
                queued_index -= 1
                launcher_count = 1
            elif running_job.priority > queued_job.priority:
                selected_job = running_job
                running_index -= 1
            else:
                selected_job = queued_job
                self.queued_jobs.pop(queued_index)
                queued_index -= 1
                launcher_count = 1

            if selected_job.replicas < selected_job.max_replicas:
                new_replicas = min(selected_job.max_replicas, 
                                   selected_job.replicas + self.free_slots - launcher_count)
                
                if new_replicas < selected_job.min_replicas:
                    if launcher_count == 1:
                        self.queued_jobs.push(selected_job)
                        #queued_index += 1
                    continue

                #print(selected_job.job_name, launcher_count, selected_job.last_action, selected_job)
                if selected_job.status == "running" and selected_job.last_action + self.T_rescale_gap > timestamp:
                    min_time = min(min_time, self.T_rescale_gap - timestamp + selected_job.last_action)
                    continue

                if selected_job.status == "expanding":
                    continue

                self.free_slots -= (new_replicas - selected_job.replicas + launcher_count)

                if launcher_count == 1:
                    self.event_heap.push(StartEvent(timestamp + DELAY, selected_job, new_replicas))
                else:
                    self.event_heap.push(RescaleEvent(timestamp + DELAY, selected_job, selected_job.replicas, new_replicas))

                if selected_job.status == "running":
                    selected_job.status = "expanding"

        if min_time < 1 << 30:
            self.event_heap.push(AssignFreeSlotsEvent(timestamp + min_time + DELAY, None))
    
    def check_replicas(self, timestamp, job):
        replicas = min(job.max_replicas, self.free_slots - 1)
        min_time = 1 << 30
        if replicas > job.min_replicas:
            return replicas
        else:
            to_free = job.min_replicas - self.free_slots + 1
            index = 0
            while to_free > 0 and index < len(self.running_jobs):
                if index == len(self.running_jobs):
                    break

                iter_job = self.running_jobs.jobs[index]
                index += 1
                if iter_job.status != "running":
                    continue
                if iter_job.priority >= job.priority:
                    break
                if iter_job.replicas > iter_job.min_replicas:
                    if iter_job.last_action + self.T_rescale_gap < timestamp:
                        new_replicas = max(iter_job.min_replicas, iter_job.replicas - to_free)
                        #self.event_heap.push(RescaleEvent(event.timestamp, job, job.replicas, new_replicas))
                        to_free -= (iter_job.replicas - new_replicas)
                        #self.free_slots += (job.replicas - new_replicas)
                    else:
                        min_time = min(min_time, iter_job.last_action + self.T_rescale_gap - timestamp)
            
            if min_time < 1 << 30:
                self.event_heap.push(CheckQueueEvent(timestamp + min_time + DELAY, None))

            if to_free > 0:
                return 0
            else:
                to_free = job.min_replicas - self.free_slots + 1
                index = 0
                while to_free > 0 and index < len(self.running_jobs):
                    if index == len(self.running_jobs):
                        break

                    iter_job = self.running_jobs.jobs[index]
                    index += 1
                    if iter_job.status != "running" or iter_job.last_action + self.T_rescale_gap > timestamp:
                        continue
                    if iter_job.priority > job.priority:
                        break
                    if iter_job.replicas > iter_job.min_replicas:
                        new_replicas = max(iter_job.min_replicas, iter_job.replicas - to_free)
                        self.event_heap.push(RescaleEvent(timestamp + DELAY, iter_job, iter_job.replicas, new_replicas))
                        iter_job.status = "expanding"
                        to_free -= (iter_job.replicas - new_replicas)
                        self.free_slots += (iter_job.replicas - new_replicas)
                return job.min_replicas
    
    def handle_submit(self, event):
        event.job.submit_time = event.timestamp
        replicas = self.check_replicas(event.timestamp, event.job)
        if replicas == 0:
            self.queued_jobs.push(event.job)
        else:
            start_event = StartEvent(event.timestamp, event.job, replicas)
            self.event_heap.push(start_event)
            self.free_slots -= (replicas + 1)
            
    def handle_start(self, event):
        #print("Free slots = %i" % self.free_slots)
        event.job.replicas = event.replicas
        runtime = event.job.get_completion_time()
        event.job.set_last_action(event.timestamp)
        event.job.status = "running"
        #print("Starting %s, last action = %f" % (event.job.job_name, event.timestamp), event.job)
        complete_event = CompleteEvent(event.timestamp + runtime + event.job.get_startup_overhead(), event.job)
        self.running_jobs.push(event.job)
        self.event_heap.push(complete_event)

    def handle_complete(self, event):
        self.free_slots += event.job.replicas + 1
        self.event_heap.remove(event.job)
        self.running_jobs.remove(event.job)
        self.assign_free_workers(event.timestamp)

    def handle_rescale(self, event):
        # remove the old complete event
        self.event_heap.remove(event.job)
        
        overhead = event.job.get_rescale_overhead()
        event.job.update_runtime(event.timestamp - event.job.last_action)

        event.job.replicas = event.new_replicas
        event.job.set_last_action(event.timestamp + overhead)
        event.job.status = "running"

        # calculate the new completion time
        new_completion_time = event.job.last_action + event.job.get_completion_time()
        # create a new complete event
        complete_event = CompleteEvent(new_completion_time, event.job)
        # push the new complete event into the heap
        self.event_heap.push(complete_event)

    def handle_assign_free(self, event):
        # check if any free slots can be assigned to any queued jobs
        if self.free_slots > 0:
            self.assign_free_workers(event.timestamp)

    def handle_check_queue(self, event):
        index = len(self.queued_jobs) - 1

        while index >= 0:
            job = self.queued_jobs.jobs[index]
            replicas = self.check_replicas(event.timestamp, job)
            if replicas > 0:
                start_event = StartEvent(event.timestamp + DELAY, job, replicas)
                self.event_heap.push(start_event)
                self.free_slots -= (replicas + 1)
                #print("Popping %s from queue %i, Free slots =" % (self.queued_jobs.jobs[index].job_name, replicas), self.free_slots)
                self.queued_jobs.pop(index)
            index -= 1

    def simulate(self, timestamps, jobs):
        for t, j in zip(timestamps, jobs):
            event = SubmitEvent(t, j)
            self.event_heap.push(event)

        event_stream = []
        while len(self.event_heap) > 0:
            event = self.event_heap.pop()
            num_replicas = 0
            for job in self.running_jobs.jobs:
                num_replicas += (job.replicas + 1)
            #print("Free slots = %i, occ = %i" % (self.free_slots, num_replicas))
            if not isinstance(event, CheckQueueEvent) and not isinstance(event, AssignFreeSlotsEvent):
                event_stream.append(deepcopy(event))
            #print(event.timestamp, event.job, event)
            if isinstance(event, SubmitEvent):
                self.handle_submit(event)
            elif isinstance(event, StartEvent):
                self.handle_start(event)
            elif isinstance(event, CompleteEvent):
                self.handle_complete(event)
            elif isinstance(event, RescaleEvent):
                self.handle_rescale(event)
            elif isinstance(event, CheckQueueEvent):
                self.handle_check_queue(event)
            elif isinstance(event, AssignFreeSlotsEvent):
                self.handle_assign_free(event)

        return event_stream
    

def plot_stacked(timestamps, utilizations, job_names, max_pes):
    plt.figure(figsize=(12, 5))
    utilization_by_job = utilizations

    #total_utilization = [sum(entry.values()) for entry in utilizations]

    #mean_utilization = sum(total_utilization) / (max_pes * len(total_utilization))
    #print("Mean utilization: ", mean_utilization * 100)

    #timestamps = [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps]
    utilizations_plt = []
    for job in job_names:
        if job in utilization_by_job:
            utilizations_plt.append(np.array(utilization_by_job[job])/float(max_pes))
    plt.stackplot(timestamps, utilizations_plt, labels=job_names)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Utilization')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.plot(timestamps, [1 for i in timestamps])
    print("Total time: ", timestamps[-1])
    #plt.savefig("stacked.pdf")
    plt.show()

def plot_utilization(event_stream, job_names, max_pes):
    final_time = event_stream[-1].timestamp
    timestamps = [2*i for i in range(1 + int((final_time + 3) // 2))]
    utilization_by_job = defaultdict(lambda: [0] * len(timestamps))
    for event in event_stream:
        #if isinstance(event, SubmitEvent):
        #    job_names.append(event.job.job_name)
        if isinstance(event, StartEvent):
            index = next(i for i, t in enumerate(timestamps) if event.timestamp < t)
            for i in range(index, len(timestamps)):
                utilization_by_job[event.job.job_name][i] = event.replicas + 1
        elif isinstance(event, RescaleEvent):
            index = next(i for i, t in enumerate(timestamps) if event.timestamp < t)
            for i in range(index, len(timestamps)):
                utilization_by_job[event.job.job_name][i] = event.new_replicas + 1
        elif isinstance(event, CompleteEvent):
            #print(event.timestamp, timestamps[-1])
            index = next(i for i, t in enumerate(timestamps) if event.timestamp < t)
            for i in range(index, len(timestamps)):
                utilization_by_job[event.job.job_name][i] = 0
    
    total_utilization = [0] * len(timestamps)
    for name, util in utilization_by_job.items():
        total_utilization = [total_utilization[i] + util[i] for i in range(len(util))]
    print("Mean utilization =", sum(total_utilization) / len(total_utilization) / max_pes)
    plot_stacked(timestamps, utilization_by_job, job_names, max_pes)
    
def get_stats(event_stream, max_pes):
    final_time = event_stream[-1].timestamp
    timestamps = [2*i for i in range(1 + int((final_time + 3) // 2))]
    utilization_by_job = defaultdict(lambda: [0] * len(timestamps))
    for event in event_stream:
        #if isinstance(event, SubmitEvent):
        #    job_names.append(event.job.job_name)
        if isinstance(event, StartEvent):
            index = next(i for i, t in enumerate(timestamps) if event.timestamp < t)
            for i in range(index, len(timestamps)):
                utilization_by_job[event.job.job_name][i] = event.replicas + 1
        elif isinstance(event, RescaleEvent):
            index = next(i for i, t in enumerate(timestamps) if event.timestamp < t)
            for i in range(index, len(timestamps)):
                utilization_by_job[event.job.job_name][i] = event.new_replicas + 1
        elif isinstance(event, CompleteEvent):
            #print(event.timestamp, timestamps[-1])
            index = next(i for i, t in enumerate(timestamps) if event.timestamp < t)
            for i in range(index, len(timestamps)):
                utilization_by_job[event.job.job_name][i] = 0
    
    total_utilization = [0] * len(timestamps)
    for name, util in utilization_by_job.items():
        total_utilization = [total_utilization[i] + util[i] for i in range(len(util))]

    job_start_times = {}
    for job_name, utilization in utilization_by_job.items():
        for i, util in enumerate(utilization):
            if util > 0:
                job_start_times[job_name] = timestamps[i]
                break

    job_end_times = {}
    for job_name, utilization in utilization_by_job.items():
        for i in range(len(utilization) - 1, -1, -1):
            if utilization[i] > 0:
                job_end_times[job_name] = timestamps[i]
                break

    job_submit_start_diff = {}
    job_priorities = {}
    for event in event_stream:
        if isinstance(event, SubmitEvent):
            job_submit_start_diff[event.job.job_name] = {"submit_time": event.timestamp}
            job_priorities[event.job.job_name] = event.job.priority
        elif isinstance(event, StartEvent):
            if event.job.job_name in job_submit_start_diff:
                job_submit_start_diff[event.job.job_name]["start_time"] = event.timestamp
        elif isinstance(event, CompleteEvent):
            if event.job.job_name in job_submit_start_diff:
                job_submit_start_diff[event.job.job_name]["end_time"] = event.timestamp

    for job_name, times in job_submit_start_diff.items():
        if "submit_time" in times and "start_time" in times and "end_time" in times:
            times["response"] = times["start_time"] - times["submit_time"]
            times["completion"] = times["end_time"] - times["submit_time"]
        else:
            times["response"] = None

    #print("Job submit-start time differences:")
    #for job_name, times in job_submit_start_diff.items():
    #    print(f"{job_name}: {times['diff']} seconds")

    total_response = 0
    total_completion = 0
    total_priority = 0
    for job_name, times in job_submit_start_diff.items():
        total_response += times['response'] * job_priorities[job_name]
        total_completion += times['completion'] * job_priorities[job_name]
        total_priority += job_priorities[job_name]

    mean_response = total_response / total_priority
    mean_completion = total_completion / total_priority

    mean_utilization = sum(total_utilization) / len(total_utilization) / max_pes
    #print(mean_utilization, mean_response)

    return final_time, mean_response, mean_completion, mean_utilization
