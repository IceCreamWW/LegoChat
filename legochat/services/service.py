class Service:
    def build_components(self, configs):
        self.components = {}
        for name, config in configs:
            component_cls = Component.get_component().get_worker_cls(f"{component['name']}")
            component = component_cls(**component.get("config", {}))
            self.components[name] = component
        return components

    def run_service(self):
        for component in self.components:
            component.start_worker()
