import numpy as np
from casadi import SX, vertcat, nlpsol, atan

class MPCController:
    def __init__(self, N=5, dt=0.1, Lf=2.0, desired_speed=10.0):
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

    def solve(self, state, coeffs):
        """
        Solves the MPC optimization problem.

        :param state: Current state [x, y, psi, v, cte, epsi]
        :param coeffs: Quadratic polynomial coefficients in vehicle coords in descending order [b2, b1, b0]
                       representing y_v(x_v) = b2*x_v² + b1*x_v + b0.
        :return: Optimal steering angle (delta) and throttle (a).
        """
        print("coeffs: ", coeffs)
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
            cost += 200 * (cte[t] ** 2)   
            cost += 200 * (epsi[t] ** 2) 
            cost += 2 * (v[t] - desired_speed) ** 2 

        for t in range(N - 1):
            cost += 1500 * (delta[t] ** 2) 
            cost += 10 * (a[t] ** 2)      

        for t in range(N - 2):
            cost += 1500 * ((delta[t + 1] - delta[t]) ** 2) 
            cost += 30 * ((a[t + 1] - a[t]) ** 2)

        constraints = []

        constraints += [x[0] - x0]
        constraints += [y[0] - y0]
        constraints += [psi[0] - psi0]
        constraints += [v[0] - v0]
        constraints += [cte[0] - cte0]
        constraints += [epsi[0] - epsi0]

        b2, b1, b0 = coeffs

        for t in range(N - 1):
            x1 = x[t + 1]
            y1 = y[t + 1]
            psi1 = psi[t + 1]
            v1 = v[t + 1]
            cte1 = cte[t + 1]
            epsi1 = epsi[t + 1]

            x0_t = x[t]
            y0_t = y[t]
            psi0_t = psi[t]
            v0_t = v[t]
            cte0_t = cte[t]
            epsi0_t = epsi[t]
            delta_t = delta[t]
            a_t = a[t]

            f0 = b2 * x0_t**2 + b1 * x0_t + b0
            psides0 = atan(2 * b2 * x0_t + b1)

            constraints += [x1 - (x0_t + v0_t * np.cos(psi0_t) * dt)]
            constraints += [y1 - (y0_t + v0_t * np.sin(psi0_t) * dt)]
            constraints += [psi1 - (psi0_t + v0_t * delta_t / Lf * dt)]
            constraints += [v1 - (v0_t + a_t * dt)]
            constraints += [cte1 - ((f0 - y0_t) + v0_t * np.sin(epsi0_t) * dt)]
            constraints += [epsi1 - ((psi0_t - psides0) + v0_t * delta_t / Lf * dt)]

        opt_vars = vertcat(x, y, psi, v, cte, epsi, delta, a)

        n_vars = opt_vars.size()[0]
        lbx = np.full(n_vars, -1e20)
        ubx = np.full(n_vars, 1e20)

        steer_start = 6 * N
        steer_end = 6 * N + (N - 1)
        lbx[steer_start:steer_end] = -0.436332  
        ubx[steer_start:steer_end] = 0.436332 

        throttle_start = steer_end
        throttle_end = throttle_start + (N - 1)
        lbx[throttle_start:throttle_end] = -1.0 
        ubx[throttle_start:throttle_end] = 1.0 

        v_start = 3 * N
        v_end = 4 * N
        lbx[v_start:v_end] = 0.0  

        max_speed = 30.0
        ubx[v_start:v_end] = max_speed  

        max_delta_change = 0.05  
        for t in range(N - 2):
            constraints += [delta[t + 1] - delta[t] <= max_delta_change]
            constraints += [delta[t + 1] - delta[t] >= -max_delta_change]

        opt_vars = vertcat(x, y, psi, v, cte, epsi, delta, a)

        n_vars = opt_vars.size()[0]
        lbx = np.full(n_vars, -1e20)
        ubx = np.full(n_vars, 1e20)

        throttle_start = steer_end
        throttle_end = throttle_start + (N - 1)
        lbx[throttle_start:throttle_end] = -1.0 
        ubx[throttle_start:throttle_end] = 1.0 

        v_start = 3 * N
        v_end = 4 * N
        lbx[v_start:v_end] = 0.0 
        ubx[v_start:v_end] = max_speed 

        lbg = np.zeros(len(constraints))
        ubg = np.zeros(len(constraints))

        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': vertcat(*constraints)
        }

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt': {'max_iter': 500}}
        solver = nlpsol('solver', 'ipopt', nlp, opts)

        x0_init = np.zeros(n_vars)

        sol = solver(
            x0=x0_init,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg
        )

        sol_values = sol['x'].full().flatten()

        delta_opt = sol_values[steer_start]
        a_opt = sol_values[throttle_start]
        return delta_opt, a_opt
