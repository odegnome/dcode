import os

if os.access('simulation/',os.F_OK):
    with os.scandir('simulation/') as scanner:
        for file in scanner:
            os.remove(file)
