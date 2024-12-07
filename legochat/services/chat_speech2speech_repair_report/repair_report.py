from typing import Dict
import rapidfuzz
import json
from data import LOCATIONS, CATEGORIES
from itn.chinese.inverse_normalizer import InverseNormalizer


normalize = InverseNormalizer().normalize

class Location:
    def __init__(self, building=None, floor=None, room=None):
        self.building = normalize(str(building)) if building else None
        self.floor = normalize(str(floor)) if floor else None
        self.room = normalize(str(room)) if room else None

    def is_empty(self):
        return self.building is None and self.floor is None and self.room is None

    def __str__(self):
        items = [self.building, self.floor, self.room]
        return ":".join([str(item) for item in items if item])

    def to_dict(self):
        return dict(building=self.building, floor=self.floor, room=self.room)


class Category:

    def __init__(self, primary=None, secondary=None):
        self.primary = primary
        self.secondary = secondary

    def is_empty(self):
        return self.primary is None and self.secondary is None

    def __str__(self):
        items = [self.primary, self.secondary]
        return ":".join([str(item) for item in items if item])

    def to_dict(self):
        return dict(primary=self.primary, secondary=self.secondary)


all_locations = [Location(**config) for config in LOCATIONS]
all_categories = [Category(**config) for config in CATEGORIES]


class RepairReportBot:
    def __init__(self):
        self.location = Location()
        self.category = Category()

    @property
    def system_prompt(self):
        return [
            {
                "role": "system",
                "content": (
                    "你是一个帮助用户报修设备的助手，你需要从用户提供的信息中解析出要报修的设备所在地点和类别。\n"
                    "以json格式回复, 地点需要包括building，floor，room三个键，类别需要包含primary，secondary两个key。\n"
                    f"类别列表：{','.join(str(category) for category in all_categories)}\n"
                ),
            },
            {"role": "assistant", "content": "请问描述您要报修的问题和位置"},
            {"role": "user", "content": "我的宿舍水龙头漏水了"},
            {
                "role": "assistant",
                "content": '{"location": {"building": "宿舍", "floor": null, "room": null}, "category": {"primary": "水", "secondary": "洗手池龙头"}}',
            },
            {"role": "assistant", "content": "请问描述您要报修的问题和位置"},
            {"role": "user", "content": "我的宿舍83号楼1201的水龙头漏水了"},
            {
                "role": "assistant",
                "content": '{"location": {"building": "宿舍83号楼", "floor": null, "room": 1201}, "category": {"primary": "水", "secondary": "洗手池龙头"}}',
            },
            {"role": "assistant", "content": "请问描述您要报修的问题和位置"},
            {"role": "user", "content": "有一个插座坏了。"},
            {"role": "user", "content": "图书馆5楼503房间"},
            {
                "role": "assistant",
                "content": '{"location": {"building": 图书馆, "floor": "5楼", "room": 503}, "category": {"primary": "电", "secondary": "插座"}}',
            },
            {"role": "assistant", "content": "请问描述您要报修的问题和位置"},
        ]

    def next_question_given_response(self, message: str):
        print(message)
        try:
            message_dict = json.loads(message)
        except:
            return self.next_question()
        self.location = Location(**message_dict["location"])
        self.category = Category(**message_dict["category"])
        return self.next_question()

    def next_question(self):
        if self.category.is_empty():
            return "请问是什么设施需要报修呢？"

        if self.location.is_empty():
            return "请问报修地点在什么位置？"

        if self.location.building is None:
            if self.location.room is None:
                return "请问是在哪个建筑的那个房间？"
            else:
                return "请问是在哪个建筑？"

        possible_buildings = [location.building for location in all_locations]
        matchest_building = rapidfuzz.process.extractOne(
            self.location.building, possible_buildings
        )[0]
        if matchest_building != self.location.building:
            return f"请问您指的是{matchest_building}吗？"

        if self.location.room is None:
            return f"请问是在{matchest_building}的哪个房间？"

        possible_rooms = list(set(
            [
                location.room
                for location in all_locations
                if location.building == self.location.building
            ]
        ))
        assert len(possible_rooms) > 0, f"no room found for building {self.location.building}"
        matchest_room = rapidfuzz.process.extractOne(
            self.location.room, possible_rooms
        )[0]

        if matchest_room != self.location.room:
            return f"请问您指的是{matchest_building}的{matchest_room}吗？"

        if self.location.room is not None:
            possible_floors = list(set(
                [
                    location.floor
                    for location in all_locations
                    if location.building == self.location.building
                    and location.room == self.location.room
                ]
            ))
            if len(possible_floors) > 1:
                return f"请问您指的是{matchest_building}, {possible_floors[0]}的{matchest_room}吗？"
        return None

    def to_dict(self):
        return dict(location=self.location.to_dict(), category=self.category.to_dict())

    def to_markdown(self):
        return f"""
- location: {self.location}
- category: {self.category}
"""


if __name__ == "__main__":
    bot = RepairReportBot()
