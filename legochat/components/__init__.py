import asyncio
import importlib
import types
from multiprocessing import Pipe, Queue
from pathlib import Path


class Component:

    async def process(self, **kwargs):
        parent_conn, child_conn = Pipe()
        self.queue.put((kwargs, child_conn))
        result = await asyncio.to_thread(parent_conn.recv)
        return result

    def setup(self):
        raise NotImplementedError

    def run(self, queue):
        self.setup()
        self.queue = queue
        while True:
            item = queue.get()
            if item is None:
                break
            process_kwargs, pipe = item
            result = self.process_func(**process_kwargs)
            pipe.send(result)

    @classmethod
    def from_config(cls, component_cls_name, params):
        queue = Queue()
        component_cls = get_component_cls(component_cls_name)
        component = component_cls(**params)
        component.queue = queue
        return component


component2cls = {}


def register_component(name):
    def wrapper(cls):
        component2cls[name] = cls
        return cls

    return wrapper


def get_component_cls(name):
    return component2cls.get(name)


def import_components(components_dir, namespace):
    for file in components_dir.glob("*.py"):
        if file.stem.startswith("__"):
            continue
        importlib.import_module(namespace + "." + file.stem)


# automatically import any Python files in the models/ directory
models_dir = Path(__file__).parent
for component in ["vad", "speech2text", "chatbot", "text2speech"]:
    import_components(models_dir / component, f"legochat.components.{component}")
