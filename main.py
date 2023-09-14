from src.cache import Cache


if __name__ == '__main__':
  a = Cache()
  b = Cache()

  a.add('foo', 1337)
  a.add('bar', 420)