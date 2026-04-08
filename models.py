from pydantic import BaseModel


class Observation(BaseModel):
    lane1: int
    lane2: int
    light: int
    emergency: int

    def as_tuple(self):
        return (self.lane1, self.lane2, self.light, self.emergency)

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.as_tuple()[index]


class Action(BaseModel):
    action: int
