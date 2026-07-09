import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym

class StandEnv(gym.Env):
    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        
        # Declare the agent and target location
        self._agent_location  = np.array([-1, -1, -1], dtype=np.float32)
        self.max_episode_steps = 1000
        self._step_count = 0
        
        self._jointArray = [i for i in range(12)]
        
        self.observation_space = gym.spaces.Dict(
            {
                "baseOrientation": gym.spaces.Box(
                    low=np.array([-np.pi, -np.pi], dtype=np.float32),
                    high=np.array([np.pi, np.pi], dtype=np.float32),
                    dtype=np.float32
                ),
                "jointAngles": gym.spaces.Box(
                    low=np.array([-np.pi for x in range(len(self._jointArray))]),
                    high=np.array([np.pi for y in range(len(self._jointArray))]),
                    dtype=np.float32
                ),
                "jointVelocities": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(12,),
                    dtype=np.float32
                ),
                "baseAngularVelocity": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32
                ),
                "relativeTargetPosition": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(12,),
            dtype=np.float32
        )

    def _get_obs(self):
        # Get and store all the observations as local variables
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation) # Used for base orientation
        joint_states = p.getJointStates(self.robot, self._jointArray) # Used for Joint Angles and joint Velocities
        jointAngles = np.array([state[0] for state in joint_states],dtype=np.float32)
        jointVelocities = np.array([state[1] for state in joint_states],dtype=np.float32)
        _, baseAngularVelocity = p.getBaseVelocity(self.robot) # Used for baseAngularVelocity

        # Return all the stored observations as one dictionary with the same naming conventions and formats as the observation_space
        return {
            "baseOrientation": np.array([roll, pitch], dtype=np.float32),
            "jointAngles": np.array(jointAngles, dtype=np.float32),
            "jointVelocities": np.array(jointVelocities, dtype=np.float32),
            "baseAngularVelocity": np.array(baseAngularVelocity, dtype=np.float32),
            "relativeTargetPosition": np.zeros(2, dtype=np.float32)
        }
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        agent_x = 0
        agent_y = 0
        
        self._agent_location = [agent_x, agent_y, 0.5]
        self._step_count = 0
        
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("laikago/laikago.urdf",self._agent_location, p.getQuaternionFromEuler([np.deg2rad(90),0,0]))

        observation = self._get_obs()
        return observation, {}
    
    def step(self, action):
        self._step_count += 1

        # REWARD CONSTANTS
        UPRIGHT = 0.1
        COMPLETION = 50.0

        reward = 0
        terminated = False
        truncated = False

        # Apply the Action
        p.setJointMotorControlArray(self.robot, self._jointArray, p.POSITION_CONTROL, action)

        # Advance the simulation
        p.stepSimulation()

        # Get Observations
        observation = self._get_obs()
        
        # Check if terminated (Robot has reached the target position / Robot has fallen over)  
        max_tilt = 45
        if abs(observation["baseOrientation"][0]) > np.deg2rad(90 + max_tilt) or abs(observation["baseOrientation"][1]) > np.deg2rad(max_tilt):
            terminated = True
            reward -= COMPLETION

        # Check if Truncated
        if self._step_count >= self.max_episode_steps:
            truncated = True

        if self._step_count >= self.max_episode_steps and terminated == False:
            reward += COMPLETION
        
        # Compute Upright Reward
        tiltThreshold = 20
        if abs(observation["baseOrientation"][0]) < np.deg2rad(90 + tiltThreshold) and abs(observation["baseOrientation"][1]) < np.deg2rad(tiltThreshold):
            reward += UPRIGHT
        elif abs(observation["baseOrientation"][0]) > np.deg2rad(90 + tiltThreshold) or abs(observation["baseOrientation"][1]) > np.deg2rad(tiltThreshold):
            reward -= UPRIGHT
        
        return observation, reward, terminated, truncated, {}
