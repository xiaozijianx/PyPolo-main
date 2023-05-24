from time import time as timer

def PrintExecutionTime(func):
    def wrapper(*args, **kwargs):
      T_1 = timer()
      ret = func(*args, **kwargs)
      T_2 = timer()
      print(func.__name__, "cost", T_2-T_1, "seconds")
      return ret
    return wrapper

def AppendExecutionTime(func):
    def wrapper(*args, **kwargs):
      T_1 = timer()
      ret = func(*args, **kwargs)
      T_2 = timer()
      return ret, T_2 - T_1
    return wrapper