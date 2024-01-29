class Foo:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def __init__(self):
        self.a = 0
        self.b = 0


class Foo:
    X = 234


print(Foo.X)
