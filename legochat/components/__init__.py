import importlib
from multiprocessing import Pipe, Queue
from pathlib import Path


class Component:

    def process(self, **kwargs):
        parent_conn, child_conn = Pipe()
        self.queue.put((kwargs, child_conn))
        result = parent_conn.recv()
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
            try:
                result = self.process_func(**process_kwargs)
                pipe.send(result)
            except Exception as e:
                import traceback

                traceback.print_exc()
                pipe.send(None)

    @classmethod
    def from_config(cls, component_type, component_name, params):
        queue = Queue()
        component_cls = get_component_cls(component_type, component_name)
        component = component_cls(**params)
        component.queue = queue
        return component


component2cls = {}


def register_component(component_type, component_name):
    def wrapper(cls):
        component2cls[f"{component_type}.{component_name}"] = cls
        return cls

    return wrapper


def get_component_cls(component_type, component_name):
    return component2cls.get(f"{component_type}.{component_name}")


def import_components(components_dir, namespace):
    for file in components_dir.glob("*.py"):
        if file.stem.startswith("__"):
            continue
        importlib.import_module(namespace + "." + file.stem)


# automatically import any Python files in the models/ directory
models_dir = Path(__file__).parent
for component in [
    "vad",
    "denoise",
    "diarization",
    "speech2text",
    "chatbot",
    "chatbot_slm",
    "text2speech",
]:
    import_components(models_dir / component, f"legochat.components.{component}")
