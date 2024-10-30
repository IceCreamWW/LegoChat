import asyncio
import logging
from enum import Enum
from typing import Callable, Dict, List


class Event:
    def __init__(self):
        self.handlers: List[Callable] = []

    async def trigger(self, *args, **kwargs):
        for handler in self.handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(*args, **kwargs)
            else:
                handler(*args, **kwargs)

    def add_handler(self, handler: Callable):
        self.handlers.append(handler)

    def remove_handler(self, handler: Callable):
        if handler in self.handlers:
            self.handlers.remove(handler)


class EventBus:
    def __init__(self):
        self.events: Dict[str, Event] = {}

    def add_event(self, name: str, event: Event):
        self.events[name] = event

    def on(self, event_name: str, handler: Callable):
        if event_name not in self.events:
            self.events[event_name] = Event()
        self.events[event_name].add_handler(handler)

    def off(self, event_name: str, handler: Callable):
        if event_name in self.events:
            self.events[event_name].remove_handler(handler)

    async def emit(self, event_name: str, *args, **kwargs):
        try:
            if event_name in self.events:
                await self.events[event_name].trigger(*args, **kwargs)
        except Exception as e:
            logging.exception(e)


class EventEnum(Enum):
    END_OF_TURN = "end_of_turn"
    INTERRUPT = "interrupt"
