import numpy as np
from casadi import SX, vertcat, nlpsol, atan
import csv
from datetime import datetime


class MPCController:
    def __init__(self, N=5, dt=0.1, Lf=2.0, desired_speed=5.0):
        """
        Initializes the MPC controller.

        :param N: Prediction horizon.
        :param dt: Time step duration.
        :param Lf: Distance between the center of mass of the vehicle and the front axle.
        :param desired_speed: Desired speed in m/s.
        """
        self.N = N
        self.dt = dt
        self.Lf = Lf
        self.desired_speed = desired_speed

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_file = f"mpc_data/control_data_{timestamp}.csv"

        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time_step", "delta", "a", "cte", "epsi", "v", "desired_speed"])  # Write header

    def save_control_data(self, step, delta, a, cte, epsi, v, desired_speed):
        """
        Saves control data to a CSV file.

        :param step: Current time step.
        :param delta: Optimal steering angle.
        :param a: Optimal throttle.
        :param cte: Cross-track error.
        :param epsi: Orientation error.
        :param v: Current vehicle speed.
        :param desired_speed: Desired vehicle speed.
        """
        with open(self.output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([step, delta, a, cte, epsi, v, desired_speed])

    def solve(self, state, coeffs):
        """
        Solves the MPC optimization problem.

        :param state: Current state [x, y, psi, v, cte, epsi]
        :param coeffs: Quadratic polynomial coefficients [b2, b1, b0].
        :return: Optimal steering angle (delta), throttle (a).
        """
        N = self.N
        dt = self.dt
        Lf = self.Lf
        desired_speed = self.desired_speed

        x0, y0, psi0, v0, cte0, epsi0 = state

        x = SX.sym('x', N)
        y = SX.sym('y', N)
        psi = SX.sym('psi', N)
        v = SX.sym('v', N)
        cte = SX.sym('cte', N)
        epsi = SX.sym('epsi', N)
        delta = SX.sym('delta', N - 1)
        a = SX.sym('a', N - 1)

        cost = 0
        for t in range(N):
            cost += 1 * (cte[t] ** 2)
            cost += 1 * (epsi[t] ** 2)
            cost += 1 * (v[t] - desired_speed) ** 2

        for t in range(N - 1):
            cost += 8 * (delta[t] ** 2)
            cost += 1 * (a[t] ** 2)
            if t < N - 2:
                cost += 1 * ((a[t + 1] - a[t]) ** 2)

        for t in range(N - 2):
            cost += 4 * ((delta[t + 1] - delta[t]) ** 2)
            cost += 3 * ((v[t + 1] - v[t]) ** 2)

        constraints = []
        constraints += [x[0] - x0, y[0] - y0, psi[0] - psi0, v[0] - v0, cte[0] - cte0, epsi[0] - epsi0]

        b2, b1, b0 = coeffs

        for t in range(N - 1):
            f0 = b2 * x[t]**2 + b1 * x[t] + b0
            psides0 = atan(2 * b2 * x[t] + b1)
            
            constraints += [x[t + 1] - (x[t] + v[t] * np.cos(psi[t]) * dt)]
            constraints += [y[t + 1] - (y[t] + v[t] * np.sin(psi[t]) * dt)]
            constraints += [psi[t + 1] - (psi[t] + v[t] * delta[t] / Lf * dt)]
            constraints += [v[t + 1] - (v[t] + a[t] * dt)]
            constraints += [cte[t + 1] - ((f0 - y[t]) + v[t] * np.sin(epsi[t]) * dt)]
            constraints += [epsi[t + 1] - ((psi[t] - psides0) + v[t] * delta[t] / Lf * dt)]

        opt_vars = vertcat(x, y, psi, v, cte, epsi, delta, a)
        nlp = {'x': opt_vars, 'f': cost, 'g': vertcat(*constraints)}

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt': {'max_iter': 500}
        }
        solver = nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = opt_vars.size()[0]
        x0_init = np.zeros(n_vars)

        lbx = np.full(n_vars, -1e20)
        ubx = np.full(n_vars, 1e20)

        lbg = np.zeros(len(constraints)) 
        ubg = np.zeros(len(constraints)) 
        
        delta_start = 6 * N
        a_start = delta_start + (N - 1)

        for i in range(N - 1):
            lbx[delta_start + i] = -0.436332
            ubx[delta_start + i] =  0.436332

        for i in range(N - 1):
            lbx[a_start + i] = -1.0
            ubx[a_start + i] =  1.0

        sol = solver(x0=x0_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        sol_values = sol['x'].full().flatten()

        delta_opt = sol_values[delta_start]    
        a_opt = sol_values[a_start]              
        cte_opt = cte0
        epsi_opt = epsi0
        v_opt = v0

        current_step = len(open(self.output_file).readlines()) - 1
        self.save_control_data(current_step, delta_opt, a_opt, cte_opt, epsi_opt, v_opt, desired_speed)
        return delta_opt, a_opt