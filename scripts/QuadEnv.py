from dm_control import composer 
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
    def sensor_values(self):
        """Sensor data including range finder and IMU"""
        range_sensors = self._entity.mjcf_model.find_all('sensor')
        return observable.MJCFFeature('sensordata', range_sensors)
    
    @composer.observable
    def orientation(self):
        """Quaternion representation of orientation"""
        atitude = self._entity.mjcf_model.find('body', 'quad')
        return observable.MJCFFeature('xquat', atitude)

class FlyWithoutCollision(composer.Task):
    def __init__(self, quad):
        self._quad = quad 
        
        # Enable quadrotor observables
        self._quad.observables.sensor_values.enabled = True 
        self._quad.observables.orientation.enabled = True

        # Arena
        self._arena = composer.Arena()
        self._task_observables = {}

    # Need to add task observable.
    # It should observe collision and if any end the episode.
    @property
    def root_entity(self):
        return self._quad

    @property
    def task_observables(self):
        return self._task_observables
    
    def get_reward(self, physics):
        return 0.0

