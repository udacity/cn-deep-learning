"""Generic base class for reinforcement learning tasks."""

from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench

class BaseTask:
    """Generic base class for reinforcement learning tasks.

    Concrete subclasses should:
    - Specify state and action spaces, initial condition, reward function.
    - Call agent on update, when new state is available, and pass back action.
    - Convert ROS messages to/from standard NumPy vectors for state and action.
    - Check for episode termination.
    """

    def __init__(self):
        """Define state and action spaces, initialize other task parameters."""
        pass

    def set_agent(self, agent):
        """Set an agent to carry out this task; to be called from update."""
        self.agent = agent

    def reset(self):
        """Reset task and return initial condition.

        Called at the beginning of each episode, including the very first one.
        Reset/initialize any episode-specific variables/counters/etc.;
        then return initial pose and velocity for next episode.

        Returns
        =======
        tuple: initial_pose, initial_force
        - initial_pose: Pose object defining initial position and orientation
        - initial_velocity: Twist object defining initial linear and angular velocity
        """
        raise NotImplementedError("{} must override reset()".format(self.__class__.__name__))

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        """Process current data, call agent, return action and done flag.

        Use current data to prepare state vector (need not use all available data);
        compute reward and check for episode termination (done flag); call agent.step()
        with state, reward, done to obtain action; pass back action, done.

        Params
        ======
        - timestamp: current time in seconds since episode started
        - pose: Pose object containing current position and orientation
        - angular_velocity: Vector3 object, current angular velocity
        - linear_acceleration: Vector3 object, current linear acceleration

        Returns
        =======
        tuple: action, done
        - action: Wrench object indicating force and torque to apply
        - done: boolean indicating whether this episode is complete
        """
        raise NotImplementedError("{} must override update()".format(self.__class__.__name__))
