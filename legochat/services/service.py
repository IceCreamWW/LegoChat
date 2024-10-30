class Service:
    def __init__(self, config):
        self.components = self.build_components(config["components"])
        self.sanity_check()
        self.sessions = []

    def build_components(self, configs):
        self.components = {}
        for name, config in configs:
            component_cls = Component.get_component().get_worker_cls(f"{component['name']}")
            component = component_cls(**component.get("config", {}))
            self.components[name] = component
        return components

    def sanity_check(self):
        for name in self.required_components:
            if name not in self.components:
                raise ValueError(f"Missing required component: {name}")

    def run_service(self):
        for component in self.components:
            component.start_worker()
