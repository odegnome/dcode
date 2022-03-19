Reference for Mujoco
====================

:Author: Rishabh Goel

:Version: 0.1

Definition list:

Bodies
  They specify *something* with mass and inertia. Think of it like a point
  where the whole mass of the system is concentrated. **Bodies** contain
  **geoms** and **sites**. Example: Human body, which consists of various
  body parts(geoms).

Geoms
  They are the physical shape(s) that the bodies possess.
  These specify the appearance and take part in collision calculation.
  Example: hands, feet of human body.

Sites
  They are a way of representing geoms but without any mass or inertia.
  Sites are used as placeholders for other entities or to show imaginary
  entities. For instance, a site may be used to represent the principal axes
  or for showing the coverage of a proximity sensor.
