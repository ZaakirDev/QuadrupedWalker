import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym

class WalkerEnv(gym.Env):
    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        
        # Declare the agent and target location
        self._agent_location  = np.array([-1, -1, -1], dtype=np.float32) 
        self.max_episode_steps = 1000 # NOTE TRY INCREASING THE MAX EPISODE STEPS
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
                "baseVelocity": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32
                ),
                "baseHeight": gym.spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(12,),
            dtype=np.float32
        )

    def _get_obs(self):
        # Get and store all the observations as local variables
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation) # Used for base orientation
        joint_states = p.getJointStates(self.robot, self._jointArray) # Used for Joint Angles and joint Velocities
        jointAngles = np.array([state[0] for state in joint_states],dtype=np.float32)
        jointVelocities = np.array([state[1] for state in joint_states],dtype=np.float32)
        _, baseAngularVelocity = p.getBaseVelocity(self.robot) # Used for baseAngularVelocity
        baseVelocity = p.getBaseVelocity(self.robot)

        # Return all the stored observations as one dictionary with the same naming conventions and formats as the observation_space
        return {
            "baseOrientation": np.array([roll, pitch], dtype=np.float32),
            "jointAngles": np.array(jointAngles, dtype=np.float32),
            "jointVelocities": np.array(jointVelocities, dtype=np.float32),
            "baseAngularVelocity": np.array(baseAngularVelocity, dtype=np.float32),
            "baseVelocity": np.array(baseVelocity[0], dtype=np.float32),
            "baseHeight": np.array([position[2]], dtype=np.float32),
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

        reward = 0
        terminated = False
        truncated = False

        # Apply the Action
        p.setJointMotorControlArray(self.robot, self._jointArray, p.POSITION_CONTROL, action)

        # Advance the simulation
        p.stepSimulation(physicsClientId=self.physicsClient)

        # Get Observations
        observation = self._get_obs()
        
        """CHECK IF THE ROBOT HAS FALLEN OVER / REACHED THE TARGET"""  
        heightThreshold = 0.3
        if observation["baseHeight"][0] < heightThreshold:
            terminated = True
            reward -= 10.0
        
        """COMPUTE THE REWARD FOR JOINT ANGULAR VELOCITY"""
        # NOTE: DONT ALLOW THE ROBOT TO HAVE AN w OF 0 Rad per sec
        microJointThreshold = 2.0
        if (np.abs(observation["jointVelocities"]) > np.array([microJointThreshold for _ in range(12)], dtype=np.float32)).any() or (np.abs(observation["jointVelocities"]) < np.array([0.2 for _ in range(12)], dtype=np.float32)).all():
            reward -= 0.1

        """COMPUTE THE PUNISHMENT FOR THE BASE HEIGHT"""
        reward -= abs((observation["baseHeight"][0]) - (0.45))/(0.45)

        """COMPUTE THE REWARD FOR POSITIVE BASE VELOCITY"""
        reward += abs(observation["baseVelocity"][0]) / 1

        """COMPUTE PUNISHMENT FOR DEVIATING TOO MUCH ON THE Y AXIS"""
        reward -= abs(observation["baseVelocity"][0]) / 1
        
        """COMPUTE THE PUNISHMENT WHEN IT TILTS"""        
        reward -= (abs(observation["baseOrientation"][0]) - np.deg2rad(90)) / 1
        
        reward -= (abs(observation["baseOrientation"][1])) / 1

        # Check if Truncated
        if self._step_count >= self.max_episode_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, {}
