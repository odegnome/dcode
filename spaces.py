import numpy as np

action_values = "0..1..0.05"
action_size = sum((1 for i in np.arange(0,1,0.05)))
obs_values = "Gyro, Acc, Sensors"
obs_size = 20

print(f"|{'Space':^20}|{'Values':^20}|{'Size':^4}|")
print(f"|{'Action':^20}|{action_values:^20}|{action_size:^4}|")
print(f"|{'Observation':^20}|{obs_values:^20}|{obs_size:^4}|")

print("\nAction Space")
for i in np.arange(0,1.05,0.05):
    print(f'{i:2.2f}', end=' ')
print()