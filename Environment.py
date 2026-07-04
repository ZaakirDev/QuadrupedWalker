import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym

class WalkerEnv(gym.Env):
    def __init__(self):
        # Setup pybullet simulation
        physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        
        # Declare the agent and target location
        self._agent_location  = np.array([-1, -1, -1], dtype=np.float32) 
        self._target_location = np.array([-1, -1, -1], dtype=np.float32)
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
                )
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
        relativeTargetPosition = np.array([self._target_location[0] - position[0], self._target_location[1] - position[1]], dtype=np.float32)

        # Return all the stored observations as one dictionary with the same naming conventions and formats as the observation_space
        return {
            "baseOrientation": np.array([roll, pitch], dtype=np.float32),
            "jointAngles": np.array(jointAngles, dtype=np.float32),
            "jointVelocities": np.array(jointVelocities, dtype=np.float32),
            "baseAngularVelocity": np.array(baseAngularVelocity, dtype=np.float32),
            "relativeTargetPosition": relativeTargetPosition
        }
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        agent_x = 0
        agent_y = 0

        target_x = 0
        target_y = 0

        while target_x == 0 or target_y == 0:
            target_x = self.np_random.uniform(-5, 5)
            target_y = self.np_random.uniform(-5, 5)
        
        self._agent_location = [agent_x, agent_y, 0.5]
        self._target_location = [target_x, target_y, 0.5]
        self._prevDistance = ((target_x)**2 + (target_y)**2)**0.5
        self._step_count = 0
        
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        self.target = p.loadURDF("r2d2.urdf", self._target_location, p.getQuaternionFromEuler([0,0,0]))
        self.robot = p.loadURDF("laikago/laikago.urdf",self._agent_location, p.getQuaternionFromEuler([np.deg2rad(90),0,0]))

        observation = self._get_obs()
        return observation, {}
    
    def step(self, action):
        self._step_count += 1

        # REWARD CONSTANTS
        DISTANCE = 1.0
        UPRIGHT = 0.1
        FAILIURE = 20.0
        COMPLETION = 20.0

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
        max_tilt = np.deg2rad(45)

        distance = np.linalg.norm(observation["relativeTargetPosition"])
        if distance < 0.5:  # pick a threshold
            terminated = True
            reward += COMPLETION
        elif abs(observation["baseOrientation"][0]) > np.deg2rad(135) or abs(observation["baseOrientation"][1]) > max_tilt:
            terminated = True
            reward -= FAILIURE

        # Check if Truncated
        if self._step_count >= self.max_episode_steps:
            truncated = True

        # Compute Distance Reward
        currentDistance = ((observation["relativeTargetPosition"][0])**2 + (observation["relativeTargetPosition"][1])**2)**0.5

        if currentDistance < self._prevDistance:
            reward += DISTANCE
        elif currentDistance > self._prevDistance:
            reward -= DISTANCE
        
        self._prevDistance = currentDistance
        
        # Compute Upright Reward
        tiltThreshold = np.deg2rad(20)
        if abs(observation["baseOrientation"][0]) < np.deg2rad(110) and abs(observation["baseOrientation"][1]) < tiltThreshold:
            reward += UPRIGHT
        elif abs(observation["baseOrientation"][0]) > np.deg2rad(110) or abs(observation["baseOrientation"][1]) > tiltThreshold:
            reward -= UPRIGHT
        
        return observation, reward, terminated, truncated, {}
