# I must say, this is working as expected, on the very first try!
from dm_control import composer, viewer
from QuadEnv import QuadEnv, FlyWithoutCollision

if __name__  == "__main__":
    quad = QuadEnv()
    task = FlyWithoutCollision(quad)
    env = composer.Environment(task)

    viewer.launch(env) 