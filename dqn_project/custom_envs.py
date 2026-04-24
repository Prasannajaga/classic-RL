from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class SimpleCartPoleEnv(gym.Env):
    """Small readable CartPole-like environment.

    Observation: [cart_x, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 pushes left, 1 pushes right.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_half_length = 0.5
        self.pole_mass_length = self.pole_mass * self.pole_half_length
        self.force_mag = 10.0
        self.dt = 0.02
        self.x_limit = 2.4
        self.angle_limit = math.radians(12)
        self.max_steps = 500

        high = np.array(
            [
                self.x_limit * 2.0,
                np.finfo(np.float32).max,
                self.angle_limit * 2.0,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(2)
        self.state = np.zeros(4, dtype=np.float32)
        self.elapsed_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=4).astype(np.float32)
        self.elapsed_steps = 0
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(float(theta))
        sintheta = math.sin(float(theta))

        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_half_length
            * (4.0 / 3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.elapsed_steps += 1

        terminated = bool(abs(x) > self.x_limit or abs(theta) > self.angle_limit)
        truncated = self.elapsed_steps >= self.max_steps
        reward = 1.0 if not terminated else 0.0
        info = {"x": float(x), "theta_degrees": math.degrees(float(theta))}

        if self.render_mode == "human":
            self.render()
        return self.state.copy(), reward, terminated, truncated, info

    def render(self) -> str | None:
        x, _, theta, _ = self.state
        message = f"x={x:+.2f} theta={math.degrees(float(theta)):+.1f} deg"
        if self.render_mode == "human":
            print(message)
            return None
        return message


class SimpleLunarLanderEnv(gym.Env):
    """Small readable LunarLander-like environment.

    Observation:
    [x, y, x_velocity, y_velocity, angle, angular_velocity, left_contact, right_contact]

    Actions:
    0 does nothing, 1 fires left side engine, 2 fires main engine, 3 fires right side engine.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.dt = 0.05
        self.gravity = -0.035
        self.main_thrust = 0.075
        self.side_thrust = 0.035
        self.turn_thrust = 0.045
        self.max_steps = 600
        self.landing_y = 0.0
        self.pad_half_width = 0.2

        high = np.array([1.5, 1.5, 2.0, 2.0, math.pi, 4.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(4)
        self.state = np.zeros(8, dtype=np.float32)
        self.elapsed_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        x = float(self.np_random.uniform(-0.25, 0.25))
        y = float(self.np_random.uniform(0.9, 1.1))
        vx = float(self.np_random.uniform(-0.03, 0.03))
        vy = float(self.np_random.uniform(-0.03, 0.0))
        angle = float(self.np_random.uniform(-0.08, 0.08))
        angular_velocity = float(self.np_random.uniform(-0.02, 0.02))
        self.state = np.array([x, y, vx, vy, angle, angular_velocity, 0.0, 0.0], dtype=np.float32)
        self.elapsed_steps = 0
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        x, y, vx, vy, angle, angular_velocity, _, _ = self.state
        fuel_penalty = 0.0

        if action == 1:
            vx += self.side_thrust
            angular_velocity += self.turn_thrust
            fuel_penalty = 0.03
        elif action == 2:
            vx += -math.sin(float(angle)) * self.main_thrust
            vy += math.cos(float(angle)) * self.main_thrust
            fuel_penalty = 0.08
        elif action == 3:
            vx -= self.side_thrust
            angular_velocity -= self.turn_thrust
            fuel_penalty = 0.03

        vy += self.gravity
        x += vx * self.dt
        y += vy * self.dt
        angle += angular_velocity * self.dt
        angular_velocity *= 0.995

        left_contact = 0.0
        right_contact = 0.0
        terminated = False
        success = False
        crash = False

        if y <= self.landing_y:
            y = self.landing_y
            left_contact = 1.0
            right_contact = 1.0
            landed_on_pad = abs(x) <= self.pad_half_width
            gentle = abs(vx) < 0.12 and abs(vy) < 0.12 and abs(angle) < 0.2
            success = landed_on_pad and gentle
            crash = not success
            terminated = True

        out_of_bounds = abs(x) > 1.2 or y > 1.4
        if out_of_bounds:
            crash = True
            terminated = True

        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_steps
        self.state = np.array(
            [x, y, vx, vy, angle, angular_velocity, left_contact, right_contact],
            dtype=np.float32,
        )

        distance_penalty = 0.6 * abs(x) + 0.4 * max(y, 0.0)
        speed_penalty = 0.35 * (abs(vx) + abs(vy))
        angle_penalty = 0.25 * abs(angle)
        reward = -distance_penalty - speed_penalty - angle_penalty - fuel_penalty
        if success:
            reward += 100.0
        elif crash:
            reward -= 100.0

        info = {"success": success, "crash": crash}
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        x, y, vx, vy, angle, _, left_contact, right_contact = self.state
        message = (
            f"x={x:+.2f} y={y:+.2f} vx={vx:+.2f} vy={vy:+.2f} "
            f"angle={math.degrees(float(angle)):+.1f} contact={int(left_contact)}/{int(right_contact)}"
        )
        if self.render_mode == "human":
            print(message)
            return None
        return message
