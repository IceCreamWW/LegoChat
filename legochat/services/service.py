from multiprocessing import Process

from legochat.components import Component


class Service:
    def __init__(self, config):
        for name in self.required_components:
            if name not in config["components"]:
                raise ValueError(f"Missing required component: {name}")
        self.components = self.build_components(config["components"])

    @property
    def required_components(self):
        return []

    def build_components(self, configs):
        self.components = {}
        for name, config in configs.items():
            component = Component.build(config["name"], config.get("params", {}))
            self.components[name] = component
        return components

    def run(self):
        for component in self.components.values():
            Process(target=component.run, args=(component.queue,)).start()
