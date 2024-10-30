import types
from multiprocessing import Pipe, Queue


class Component:

    async def process(self, *args, **kwargs):
        parent_conn, child_conn = Pipe()
        self.queue.put((kwargs, child_conn))
        result = await parent_conn.recv()
        return result

    def run(self, queue):
        self.queue = queue
        while True:
            item = queue.get()
            if item is None:
                break
            process_kwargs, pipe = item
            result = self.process_func(**process_kwargs)
            pipe.send(result)


def build_and_run_component(cls, params, pipe):
    queue = Queue()
    component = cls(**params)
    component.queue = queue
    Process(target=component.run, args=(queue,)).start()
    return


component2cls = {}


def register_component(name):
    def wrapper(cls):
        component2cls[name] = cls
        return cls

    return wrapper


def get_component_cls(name):
    return component2cls.get(name)
