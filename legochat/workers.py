worker2queue = {}


def get_worker_queue(worker_name):
    if worker_name not in worker2queue:
        return None
    return worker2queue[worker_name]
