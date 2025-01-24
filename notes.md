## 03-05
**Seed** will be 103743 for all experiments.

- 0-1000
- 1000-2000

Trained for 1000 episodes using uniform(0.0, 1.0) noise. After some while,
again trained for 1000 episodes totalling 2000 episodes.

critic_lr = 0.002
actor_lr = 0.001

## 05-05

> Eventhough the training is on 05-05, the results are stored in 03-05

- 2000-4000
- 4000-6000

Changed the episode initialization from preset intitial states to random
settings for `qpos` and `qvel`.Trained for 2000 episodes.

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 2.0)|
|z|~U(0.5, 4.5)|
|qvel|~N(0.0, 1.0)|

- 6000-8000

After training till 6000 episodes, changed the qvel noise to N(0,0, 0.5).
Also changed the learning rates for critic and actor to 0.02 and 0.01,
respectively.

> Maybe the low learning rate is hurting the learning process.

- 8000-10000

Changed the randomization to the following settings after 8000 episodes.

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 0.01)|
|z|~U(2, 4)|
|qvel|~N(0.0, 0.01)|
|noise|~N(0.5, 0.1)|

- 10000-12000
- 12000-14000

>No more negative reward for norm < 0.740. This is to promote manuvering
>for collision prevention. Example, if the quadrotor needs to tilt to go
>away from the wall.

Discarded the above formulation because the reward started going toward -5000, whereas
earlier it was toward -3000, which is not acceptable. Furthermore, added code to save the
optimizer weights. Should help in future episodes but not in this.

- 14000-16000

Lowered `critic_lr` and `actor_lr` to 0.005. This is because the episode reward
started at about -3000 but moved towards -5000, which might be due to too much learning
during exploration.

- 16000-18000

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 0.01)|
|z|~U(2, 4)|
|qvel|~N(0.0, 0.01)|
|noise|~N(0.0, 0.01)|

Changed `noise` added to rotor values.

- 18000-20000

Learning rate is again changed, to 0.002 and 0.001 for critic and actor.

- 20000-21000

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 0.01)|
|z|~U(2, 4)|
|qvel|~N(0.0, 0.01)|
|noise|~N(0.5, 0.01)|

## 08-05

- 10000-14000

Tried to retrain the model from trained model of 10000 episdoes. Learning are
0.002 for both actor and critic. Action noise is N(0.5, 0.01).

- 11000-14000

Trained model for 11000 episodes is used for further training with some fundamental
changes.

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 0.1)|
|z|~U(2, 4)|
|qvel|~N(0.0, 0.1)|
|noise|~N(0.1, 0.1)|

## 24-05
- 10000-20000

Current settings are: `critic_lr` is 0.002, `actor_lr` is 0.001, action noise is
N(0.0, 0.1, size=4).

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 1.0)|
|z|~U(2, 4)|
|qvel|~N(0.0, 1.0)|

- 20000-25000
Changed the following: action noise is N(0.0, 0.3, size=4)

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 0.5)|
|z|~U(2, 4)|
|qvel|~N(0.0, 1.0)|

# Testing

|Values|Distribution|
|:----:|:----------:|
|(x,y)|~N(0.0, 0.5)|
|z|~U(2, 4)|
|qvel|~N(0.0, 0.5)|
