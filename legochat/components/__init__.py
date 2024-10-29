import types


class Component:

    def process(self):
        raise NotImplementedError

    def start_worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            params, pipe = item
            result = self.process(**params)
            if isinstance(result, types.GeneratorType):
                for item in result:
                    pipe.send(item)
            else:
                pipe.send(result)


component2cls = {}


def register_component(name):
    def wrapper(cls):
        component2cls[name] = cls
        return cls

    return wrapper


def get_component_cls(name):
    return component2cls.get(name)
