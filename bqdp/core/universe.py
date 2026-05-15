# bqdp/core/universe.py





class Universe:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        
        self.n = n
        self.items = list(range(1, n + 1))

    def validate(self, item_id: int) -> bool:
        return 1 <= item_id <= self.n

    def to_dict(self) -> dict:
        return {"n": self.n}

    @classmethod
    def from_dict(cls, data: dict) -> "Universe":
        return cls(data["n"])