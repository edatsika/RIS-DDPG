import numpy as np


class RIS_DDPG(object):
    def __init__(self,
                 num_RIS,
                 num_RIS_elements,
                 num_users,
                 power_t,
                 AWGN_var,
                 bandwidth,
                 channel_est_error=False,
                 channel_noise_var=1e-2,
                 ):

        self.M = num_RIS
        self.N = num_RIS_elements # per RIS
        self.K = num_users

        self.channel_est_error = channel_est_error

        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        self.bandwidth = bandwidth

        power_t_W = 10 ** (power_t / 10) * 1e-3
        #print("power_t:", power_t)
        #print("power_t_W:", power_t_W)

        # Define the variables
        self.h_km = np.random.rand(self.K, self.M, self.N)  # Example channel matrix
        self.g_km = np.random.rand(self.K, self.M, self.N)  # Example channel matrix

        rho_size = self.K
        h_size = 2*self.K*self.M*self.N
        g_size = 2*self.K*self.M*self.N
        a_size = self.K*self.M #assume complex but actually binary
        theta_size = 2*self.K*self.M*self.N

        #self.state_dim = rho_size + h_size + g_size + a_size + theta_size # inside arrays complex number =  1 element

        self.state_dim = h_size + g_size + 2*a_size
        #print("state_dim in env:", self.state_dim)

        self.action_dim = rho_size + theta_size
        #print("action_dim in env:", self.action_dim)

        # Add parameters to optimize here
        # Create random imaginary parts in the interval [0, 2π)
        imaginary_part = np.random.uniform(0, 2 * np.pi, size=(self.K, self.M, self.N))

        # Generate the real part with values of either -1 or 1
        real_part = np.random.choice([-1, 1], size=(self.K, self.M, self.N))

        # Combine real and imaginary parts into a complex array
        self.theta_kmn = real_part + 1j * imaginary_part
        #print(self.theta_kmn.shape)

        # Create random imaginary parts in the interval [0, 2π)
        #imaginary_part = np.random.uniform(0, 2 * np.pi, size=(self.K, self.M, self.N))
        # Generate the real part with values of either -1 or 1
        #real_part = np.random.choice([-1, 1], size=(self.K, self.M, self.N))
        # Combine real and imaginary parts into a complex array
        #self.theta_kmn = real_part + 1j * imaginary_part

        #a_km_bin = np.random.randint(2, size=(self.K, self.M))
        
        self.a_km = np.zeros((self.K, self.M), dtype=int)

        for i in range(self.K):
            # Randomly select one RIS element to associate with the user
            chosen_RIS = np.random.choice(self.M)
            self.a_km[i, chosen_RIS] = 1

         # Encode binary values to real numbers
        #self.a_km = 2 * a_km_bin - 1  # Map 0 to -1 and 1 to 1
        #print(self.a_km)
        #input("self.a_km printed - Press Enter to continue...")
        #self.rho_k = np.zeros(self.K, dtype=float)

        #self.rho_k = np.random.uniform(0, powert_t_W, size=(self.K,))
        self.rho_k_min = 0.0001  # Minimum allowable transmission power (adjust as needed)
        self.rho_k = np.random.uniform(self.rho_k_min, power_t_W, size=(self.K,))

        #print("Dimensions of self.a_km:", self.a_km.shape)
        #print("Dimensions of self.rho_k:", self.rho_k.shape)
        #print("Dimensions of self.theta_kmn:", self.theta_kmn.shape)
        #print("Dimensions of self.h_km:", self.h_km.shape)
        #print("Dimensions of self.g_km:", self.g_km.shape)

        self.state = None
        self.done = None

        self.episode_t = None

    def reset(self):
        self.episode_t = 0

        # Define the initial action vector by stacking the variables
        init_action_theta = np.hstack((np.real(self.theta_kmn.reshape(1, -1)), np.imag(self.theta_kmn.reshape(1, -1))))
        #init_action = np.hstack((np.array(self.rho_k), self.a_km.reshape(-1), theta_kmn_real, theta_kmn_imag))
        #init_action = np.hstack((self.rho_k, self.a_km.reshape(-1), theta_kmn_real, theta_kmn_imag))
        #init_action = np.hstack((self.rho_k, self.a_km.reshape(-1)))
        #print("######################## ", init_action_theta.shape)
        #print("######################## ", self.rho_k.shape)
        init_action = np.hstack((self.rho_k.reshape(1, -1), init_action_theta))

        self.h_km = np.random.normal(0, np.sqrt(0.5), (self.K, self.M, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                               (self.K, self.M, self.N))
        self.g_km = np.random.normal(0, np.sqrt(0.5), (self.K, self.M, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                               (self.K, self.M, self.N))

        # Flatten the complex matrices and separate real and imaginary parts
        h_kmn_real = np.real(self.h_km).reshape(1,-1)
        h_kmn_imag = np.imag(self.h_km).reshape(1,-1)

        g_kmn_real = np.real(self.g_km).reshape(1,-1)
        g_kmn_imag = np.imag(self.g_km).reshape(1,-1)

        # Define self.state by concatenating all variables together
        #print("Input init_action in reset of env before stack", init_action.shape)
        #print("Init_action: ", init_action)
        #print("a_km: ", self.a_km)
        #print(theta_kmn_real.shape)
        #print(theta_kmn_imag.shape)
        #print(h_kmn_real.shape)
        #print(h_kmn_imag.shape)
        #print(g_kmn_real.shape)
        #print(g_kmn_imag.shape)

        sigma = self.awgn_var  # Example noise variance
        #print("rho_k_values inside step inside compute reward:", self.rho_k)
        #print("Shape of self.rho_k:", self.rho_k.shape)
        #print("Shape of self.theta_kmn:", self.theta_kmn.shape)
        # Calculate the SNR for each user and RIS
        snr_km = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                # Calculate the SNR for user k and RIS m
                a_km_values = self.a_km[k, m]  # Get the value from the binary matrix a_km
                #rho_k_values = np.array(self.rho_k)  # Convert to numpy array if not already
                #rho_k_values = self.rho_k[0,k]    # Get the transmission power for user k
                rho_k_values = self.rho_k[np.newaxis, :]
                if np.ndim(rho_k_values) >= 2:
                # Convert N-D array to 1D array
                    rho_k_values = np.squeeze(rho_k_values)
                #print("Shape of rho_k_values:", rho_k_values.shape)
                #rho_k_values = self.rho_k    # Get the transmission power for user k
                #print("rho_k_values:", rho_k_values[np.newaxis, :])
                #print("rho_k_values [k]:", rho_k_values[k])
                #print("Shape of rho_k_values:", rho_k_values.shape)
                h_km_values = self.h_km[k, m, :]  # Get the complex channel matrix h for user k and RIS element m (all N elements)
                g_km_values = self.g_km[k, m, :]  # Get the complex channel matrix g for user k and RIS element m (all N elements)

                # Calculate the SNR for user k and RIS element m (considering all N elements)
                decoded_a_km_values = a_km_values
                #decoded_a_km_values = (a_km_values + 1) / 2
                #print(k)
                #snr_km[k, m] = decoded_a_km_values * (np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values) * rho_k_values[k]) ** 2 / sigma ** 2)
                snr_km[k, m] = decoded_a_km_values * (rho_k_values[k]*np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values)) ** 2 / sigma ** 2)

        
        #self.state = np.hstack((init_action, self.a_km.reshape(1,-1), h_kmn_real, h_kmn_imag, g_kmn_real, g_kmn_imag))
        self.state = np.hstack((snr_km.reshape(1,-1), self.a_km.reshape(1,-1), h_kmn_real, h_kmn_imag, g_kmn_real, g_kmn_imag))
        #print(f"???????????????? State in reset of env: {self.state}")
        #print(f"???????????????? State shape in reset of env: {self.state.shape}")

        return self.state

    def _compute_reward(self, Pmax):
        reward = 0
        opt_reward = 0
        # Rate constraint for K users
        r_k_min = np.full((self.K, self.M), 5*self.bandwidth) # 5 bps/Hz*bandwidth
        powert_t_W = 10 ** (Pmax / 10) * 1e-3 #10 ** (self.power_t / 10) 10 ** (dbm / 10) * 1e-3

        sigma = self.awgn_var  # Example noise variance
        #print("rho_k_values inside step inside compute reward:", self.rho_k)
        #print("Shape of self.rho_k:", self.rho_k.shape)
        #print("Shape of self.theta_kmn:", self.theta_kmn.shape)
        # Calculate the SNR for each user and RIS
        snr_km = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                # Calculate the SNR for user k and RIS m
                a_km_values = self.a_km[k, m]  # Get the value from the binary matrix a_km
                rho_k_values = self.rho_k[0,k]    # Get the transmission power for user k
                h_km_values = self.h_km[k, m, :]  # Get the complex channel matrix h for user k and RIS element m (all N elements)
                g_km_values = self.g_km[k, m, :]  # Get the complex channel matrix g for user k and RIS element m (all N elements)

                # Calculate the SNR for user k and RIS element m (considering all N elements)
                decoded_a_km_values = a_km_values
                #decoded_a_km_values = (a_km_values + 1) / 2
                snr_km[k, m] = decoded_a_km_values * (rho_k_values*np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values)) ** 2 / sigma ** 2)
                #snr_km[k, m] = decoded_a_km_values * (np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values) * rho_k_values) ** 2 / sigma ** 2)
                #print("snr_km[k, m]:", snr_km[k, m])
                #print("snr_km[k, m]:", snr_km[k, m])
                #print("rho_k_values:", rho_k_values)
                #print("a_km_values:", a_km_values)
                #print("decoded_a_km_values:", decoded_a_km_values)
                #print("self.theta_kmn[k, m, :]:", self.theta_kmn[k, m, :])
                #print("h_km_values:", h_km_values)
                #print("g_km_values:", g_km_values)
        #print("a_km_values inside compute_reward:", self.a_km)
        
        # Calculate the sum rate
        sum_rate = self.bandwidth*np.sum(np.log2(1 + snr_km))
        #print("Sum_rate:", sum_rate)
        #reward = sum_rate
        #print("Current reward:", reward)
        #print("np.sum(rho_k_values):", np.sum(rho_k_values))

        # Reward with penalty
        # Check if each user's data rate meets the minimum rate requirement
        #sum_rate_k =self.bandwidth*np.log2(1 + snr_km)
        #positive_elements = sum_rate_k[sum_rate_k > 0]
        rate_meets_requirement = (sum_rate[sum_rate > 0] >= r_k_min[sum_rate > 0]).all()
        power_meets_requirement = (rho_k_values[rho_k_values > 0] <= powert_t_W).all()
        #print("sum_rate_k:", sum_rate_k/1000000)
        #print("sum_rate:", sum_rate/1000000)
        #print("r_k_min:", r_k_min)
        #print("rate_meets_requirement):", rate_meets_requirement)     
        #if rate_meets_requirement and np.sum(rho_k_values) <= powert_t_W:
        if rate_meets_requirement and power_meets_requirement:
            reward = sum_rate/1000000  # Positive reward for achieving the goal
            #print("sum_rate:", sum_rate/1000000)
        else:
            reward = -1.0  # Negative reward for not meeting the requirements

        # Calculate the reward while considering rate constraints
        #reward = sum_rate - np.sum(np.maximum(0, r_k_min - sum_rate))
        #print("Current reward:", reward)

        opt_reward = reward # example comparison: optimal reward would be without interference
        return reward, opt_reward

    def step(self, action, power_t):
        self.episode_t += 1

        # Separate the action into different components
        rho_k = action[0,:self.K].copy()  # Extract and make a copy of transmission power values
        Pmax = 10 ** (power_t / 10) * 1e-3
        #print("----------------")
        while np.sum(rho_k) > Pmax or (rho_k < 0).any():
            rho_k = np.random.uniform(1e-4, Pmax, size=(self.K,))
        #while np.sum(rho_k) > Pmax:
        # If the sum exceeds Pmax, resample rho_k
        #     rho_k = np.random.uniform(0.1, Pmax, size=(self.K,))
       # print(rho_k)

        # Update the transmission power values in the action array ?????????????????????????
        action[0, :self.K] = rho_k.reshape(1, self.K)


        #Extract theta_kmn values from action --- check if the same in Actor's forward!!!!!!!!!!!! ARE INDEXES CORRECT????
        theta_kmn_real = action[:, self.K:self.K+self.K*self.M*self.N]
        theta_kmn_imag = action[:, self.K*self.M*self.N:2*self.K*self.M*self.N]
        #print(action)
        #print(theta_kmn_real)
        #print(theta_kmn_imag)
        #input("Press Enter to continue...")
        
        # Update the theta_kmn values in the action array
        action[:, self.K:self.K+self.K*self.M*self.N] = theta_kmn_real.reshape(1, -1)
        action[:, self.K*self.M*self.N:2*self.K*self.M*self.N] = theta_kmn_imag.reshape(1, -1)
        #action[0, self.K:self.K+self.K*self.M] = theta_kmn.reshape(1, -1)
        #print(f">>>>>>>>>>>>>>>>>>>>>> Inside env.step: Action shape {action.shape}, Action: {action}")

        # Flatten the complex matrices and separate real and imaginary parts
        
        #theta_kmn_real = np.real(self.theta_kmn).reshape(1,-1)
        #theta_kmn_imag = np.imag(self.theta_kmn).reshape(1,-1)

        a_km = self.a_km.reshape(1,-1)

        h_kmn_real = np.real(self.h_km).reshape(1,-1)
        h_kmn_imag = np.imag(self.h_km).reshape(1,-1)

        g_kmn_real = np.real(self.g_km).reshape(1,-1)
        g_kmn_imag = np.imag(self.g_km).reshape(1,-1)

        sigma = self.awgn_var  # Example noise variance
        #print("rho_k_values inside step inside compute reward:", self.rho_k)
        #print("Shape of self.rho_k:", self.rho_k.shape)
        #print("Shape of self.theta_kmn:", self.theta_kmn.shape)
        # Calculate the SNR for each user and RIS
        snr_km = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                # Calculate the SNR for user k and RIS m
                a_km_values = self.a_km[k, m]  # Get the value from the binary matrix a_km
                rho_k_values = self.rho_k    # Get the transmission power for user k
                if np.ndim(rho_k_values) >= 2:
                # Convert N-D array to 1D array
                    rho_k_values = np.squeeze(rho_k_values)
                h_km_values = self.h_km[k, m, :]  # Get the complex channel matrix h for user k and RIS element m (all N elements)
                g_km_values = self.g_km[k, m, :]  # Get the complex channel matrix g for user k and RIS element m (all N elements)

                # Calculate the SNR for user k and RIS element m (considering all N elements)
                decoded_a_km_values = a_km_values
                #decoded_a_km_values = (a_km_values + 1) / 2
                #snr_km[k, m] = decoded_a_km_values * (np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values) * rho_k_values[k]) ** 2 / sigma ** 2)
                snr_km[k, m] = decoded_a_km_values * (rho_k_values[k]*np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values)) ** 2 / sigma ** 2)


        self.state = np.hstack((snr_km.reshape(1,-1), a_km, h_kmn_real, h_kmn_imag, g_kmn_real, g_kmn_imag))

        #print(f"############ Inside env.step: State shape {self.state.shape}, State: {self.state[:, :self.K]}")

        #Update env arrays for rho, a_km, theta, h, g ??
        self.rho_k = np.array(action[:, :self.K])
        #self.a_km=a_km
        #self.a_km = action[:, self.K:self.K + a_km_size]

        self.theta_kmn = theta_kmn_real + 1j * theta_kmn_imag
        # Convert it to size (K, M, N)
        theta_kmn_reshaped = self.theta_kmn.reshape(self.K, self.M * self.N).reshape(self.K, self.M, self.N)
        self.theta_kmn = theta_kmn_reshaped
        #print(self.theta_kmn.shape)
        #print(self.theta_kmn.shape)
        #print(self.theta_kmn)
        #input("Check shapes - Press Enter to continue...")
        
        reward, opt_reward = self._compute_reward(Pmax)

        done = opt_reward == reward

        return self.state, reward, done, None
    
    def close(self):
        pass
