# bqdp/core/universe.py





class Universe:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = n
        self.items = range(1, n + 1)


    @property
    def embedding_size(self) -> int:
        # index 0 is reserved as the padding symbol of the sequence encoder
        return self.n + 1


    def validate(self, item_id: int) -> bool:
        return 1 <= item_id <= self.n


    def require(self, item_id: int) -> int:
        if not self.validate(item_id):
            raise ValueError(f"{item_id} is outside universe range [1, {self.n}]")
        
        return item_id