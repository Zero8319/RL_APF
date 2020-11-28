import numpy as np
import APF_function_for_DQN
from matplotlib import pyplot as plt


class environment():
    def __init__(self):
        self.num_action = 8
        self.num_state = 4
        self.t = 0
        self.v = 300
        self.delta_t = 0.5
        self.wall_following = False
        self.rng = np.random.default_rng(0)

    def reset(self):
        self.t = 0
        self.boundary = APF_function_for_DQN.generate_boundary(np.array([[0.0], [0]]), np.array([[3500], [0]]),
                                                               np.array([[3500], [3500]]), np.array([[0], [3500]]))

        self.obstacle_point = self.rng.random((2, 3)) * 2500 + 500
        self.obstacle = APF_function_for_DQN.generate_obstacle(self.obstacle_point[:, 0:1], self.obstacle_point[:, 1:2],
                                                               self.obstacle_point[:, 2:3])
        self.obstacle_total = np.hstack((self.boundary, self.obstacle))
        while True:
            self.target_position = self.rng.random((2, 1)) * 3500
            if APF_function_for_DQN.is_target_in_obstacle(self.obstacle_point[:, 0:1], self.obstacle_point[:, 1:2],
                                                          self.obstacle_point[:, 2:3], self.target_position):
                pass
            else:
                break
        while True:
            self.agent_position = self.rng.random((2, 1)) * 3500
            if APF_function_for_DQN.is_target_in_obstacle(self.obstacle_point[:, 0:1], self.obstacle_point[:, 1:2],
                                                          self.obstacle_point[:, 2:3], self.agent_position):
                pass
            else:
                break
        self.agent_orientation = np.array([[0], [1]])
        temp = np.argmin(np.linalg.norm(self.obstacle_total - self.agent_position, axis=0))
        self.obstacle_closest = self.obstacle_total[:, temp:temp + 1]
        temp1 = self.obstacle_closest - self.agent_position
        temp2 = self.target_position - self.agent_position
        self.state = np.array([[np.linalg.norm(temp1),
                                np.arctan2(temp1[1], temp1[0]),
                                np.linalg.norm(temp2),
                                np.arctan2(temp2[1], temp2[0])]], dtype='float32')
        return self.state

    def reward(self):
        temp1 = APF_function_for_DQN.is_collision(self.obstacle_point[:, 0:1], self.obstacle_point[:, 1:2],
                                                  self.agent_position, self.last_agent_position)
        temp2 = APF_function_for_DQN.is_collision(self.obstacle_point[:, 0:1], self.obstacle_point[:, 2:3],
                                                  self.agent_position, self.last_agent_position)
        temp3 = APF_function_for_DQN.is_collision(self.obstacle_point[:, 1:2], self.obstacle_point[:, 2:3],
                                                  self.agent_position, self.last_agent_position)
        if np.linalg.norm(self.agent_position - self.target_position) < 150:
            return 0, True
        elif any([bool(self.agent_position[0] < 0), bool(self.agent_position[0] > 3500),
                  bool(self.agent_position[1] < 0), bool(self.agent_position[1] > 3500), temp1, temp2, temp3]):
            return -100, True
        elif self.t == 100:
            return -1, True
        else:
            return -1, False

    def step(self, action):
        self.t += 1
        self.last_agent_position = self.agent_position
        if action == 0:
            scale_repulse = 0
        elif action == 1:
            scale_repulse = 10000000
        elif action == 2:
            scale_repulse = 20000000
        elif action == 3:
            scale_repulse = 50000000
        elif action == 4:
            scale_repulse = 100000000
        elif action == 5:
            scale_repulse = 200000000
        elif action == 6:
            scale_repulse = 500000000
        elif action == 7:
            scale_repulse = 800000000
        F, F_attract, F_repulse, self.wall_following = APF_function_for_DQN.total_decision(self.agent_position,
                                                                                           self.agent_orientation,
                                                                                           self.obstacle_total,
                                                                                           self.target_position,
                                                                                           scale_repulse)
        self.agent_position = self.agent_position + F / np.linalg.norm(F) * self.v * self.delta_t
        self.agent_orientation = F
        temp = np.argmin(np.linalg.norm(self.obstacle_total - self.agent_position, axis=0))
        self.obstacle_closest = self.obstacle_total[:, temp:temp + 1]
        temp1 = self.obstacle_closest - self.agent_position
        temp2 = self.target_position - self.agent_position
        self.state = np.array([[np.linalg.norm(temp1),
                                np.arctan2(temp1[1], temp1[0]),
                                np.linalg.norm(temp2),
                                np.arctan2(temp2[1], temp2[0])]], dtype='float32')
        reward, done = self.reward()
        return self.state, reward, done

    def render(self):
        plt.cla()
        plt.plot(self.obstacle[0, :], self.obstacle[1, :], 'blue')
        plt.plot(self.boundary[0, :], self.boundary[1, :], 'blue')
        plt.scatter(self.agent_position[0], self.agent_position[1], c='black')
        plt.scatter(self.target_position[0], self.target_position[1], c='red')
        if self.wall_following:
            plt.quiver(self.agent_position[0], self.agent_position[1], self.agent_orientation[0],
                       self.agent_orientation[1], color='green')
        else:
            plt.quiver(self.agent_position[0], self.agent_position[1], self.agent_orientation[0],
                       self.agent_orientation[1], color='black')
        plt.show(block=False)
        plt.pause(0.01)
