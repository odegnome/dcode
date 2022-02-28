from dm_control import mujoco, composer 
from dm_control import mjcf
from dm_control.composer.observation import observable

class QuadEnv(composer.Entity):
    def _build(self):
        self._model = mjcf.from_path('quad.xml')
    
    def _build_observables(self):
        return QuadObservables(self)
    @property
    def mjcf_model(self):
        return self._model
    
    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))

class QuadObservables(composer.Observables):
    """Adds Sensor data and current actuator values as observables"""

    @composer.observable
    def rotor_values(self):
        """Actuator control values"""
        rotors = self._entity.mjcf_model.find_all('actuator')
        return observable.MJCFFeature('actuator_force', rotors)

    @composer.observable
    def sensor_values(self):
        """Sensor data including range finder and IMU"""
        range_sensors = self._entity.mjcf_model.find_all('sensor')
        return observable.MJCFFeature('sensordata', range_sensors)
