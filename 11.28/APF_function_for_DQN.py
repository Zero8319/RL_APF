import numpy as np


def attract(self_position, target_position):
    F = (target_position - self_position) / np.linalg.norm(target_position - self_position)
    return F


def repulse(self_position, obstacle_position, influence_range, scale):
    F = scale * (1 / np.linalg.norm(self_position - obstacle_position) - 1 / influence_range) / np.linalg.norm(
        self_position - obstacle_position) ** 2 * (self_position - obstacle_position) / np.linalg.norm(
        self_position - obstacle_position)

    if np.linalg.norm(self_position - obstacle_position) < influence_range:
        return F
    else:
        return np.array([[0], [0]])


def generate_boundary(point1, point2, point3, point4):
    boundary12 = np.vstack(
        (np.ravel(np.linspace(point1[0], point2[0], 50)), np.ravel(np.linspace(point1[1], point2[1], 50))))
    boundary23 = np.vstack(
        (np.ravel(np.linspace(point2[0], point3[0], 50)), np.ravel(np.linspace(point2[1], point3[1], 50))))
    boundary34 = np.vstack(
        (np.ravel(np.linspace(point3[0], point4[0], 50)), np.ravel(np.linspace(point3[1], point4[1], 50))))
    boundary41 = np.vstack(
        (np.ravel(np.linspace(point4[0], point1[0], 50)), np.ravel(np.linspace(point4[1], point1[1], 50))))

    boundary = np.hstack((boundary12, boundary23, boundary34, boundary41))
    return boundary


def generate_obstacle(point1, point2, point3):
    boundary12 = np.vstack(
        (np.ravel(np.linspace(point1[0], point2[0], 50)), np.ravel(np.linspace(point1[1], point2[1], 50))))
    boundary23 = np.vstack(
        (np.ravel(np.linspace(point2[0], point3[0], 50)), np.ravel(np.linspace(point2[1], point3[1], 50))))
    boundary31 = np.vstack(
        (np.ravel(np.linspace(point3[0], point1[0], 50)), np.ravel(np.linspace(point3[1], point1[1], 50))))

    boundary = np.hstack((boundary12, boundary23, boundary31))
    return boundary


def wall_follow(self_position, obstacle, self_orientation):
    temp = np.linalg.norm(obstacle - self_position, axis=0)
    temp = np.argsort(temp)
    obstacle_position = obstacle[:, temp[0]:temp[0] + 1]
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, self_position - obstacle_position)
    rotate_vector2 = -1 * rotate_vector1
    temp1 = np.linalg.norm(rotate_vector1 - self_orientation)
    temp2 = np.linalg.norm(rotate_vector2 - self_orientation)
    if temp1 > temp2:
        return rotate_vector2
    else:
        return rotate_vector1


def APF_decision(self_position, target_position, obstacle, scale_repulse):
    influence_range = 800
    F_attract = attract(self_position, target_position)
    temp = np.linalg.norm(obstacle - self_position, axis=0)
    temp = np.argsort(temp)
    obstacle_position = obstacle[:, temp[0]:temp[0] + 1]
    F_repulse = repulse(self_position, obstacle_position, influence_range, scale_repulse)
    F = F_attract + F_repulse
    return F_attract, F_repulse, F


def total_decision(self_position, self_orientation, obstacle, target_position, scale_repulse):
    F_attract, F_repulse, F = APF_decision(self_position, target_position, obstacle, scale_repulse)
    vector1 = np.ravel(F_attract + F_repulse)
    vector2 = np.ravel(F_attract)

    if np.dot(vector1, vector2) < 0:
        F = wall_follow(self_position, obstacle, self_orientation)
        return F, F_attract, F_repulse, True

    else:
        return F, F_attract, F_repulse, False


def is_target_in_obstacle(A, B, C, target):
    temp = np.hstack((np.hstack((A, B, C)).T, np.ones((3, 1))))
    S_abc = 0.5 * np.abs(np.linalg.det(temp))
    temp = np.hstack((np.hstack((A, B, target)).T, np.ones((3, 1))))
    S_abt = 0.5 * np.abs(np.linalg.det(temp))
    temp = np.hstack((np.hstack((target, B, C)).T, np.ones((3, 1))))
    S_bct = 0.5 * np.abs(np.linalg.det(temp))
    temp = np.hstack((np.hstack((A, target, C)).T, np.ones((3, 1))))
    S_act = 0.5 * np.abs(np.linalg.det(temp))
    if np.abs((S_abc - S_abt - S_act - S_bct) / S_abc) < 0.01:
        return True
    else:
        return False


def is_collision(A, B, C, D):
    ca = np.ravel(A - C)
    cd = np.ravel(D - C)
    cb = np.ravel(B - C)
    ba = np.ravel(A - B)
    bc = np.ravel(C - B)
    bd = np.ravel(D - B)
    if np.dot(np.cross(ca, cd), np.cross(cd, cb)) > 0 and np.dot(np.cross(bc, ba), np.cross(ba, bd)) > 0:
        return True
    else:
        return False