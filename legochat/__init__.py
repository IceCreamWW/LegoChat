import multiprocessing

worker2queue = {}


def get_worker_queue(worker_name, start=True):
    if worker_name not in worker2queue:
        worker2queue[worker_name] = []

    if not worker2queue[worker_name]:
        if start:
            worker2queue[worker_name] = multiprocessing.Queue()

    return workers[worker_id]
