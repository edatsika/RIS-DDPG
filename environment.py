import numpy as np

class RIS_DDPG(object):
    def __init__(self, num_RIS, num_RIS_elements, num_users, power_t, AWGN_var, bandwidth,
                 channel_est_error=False, channel_noise_var=1e-2):

        self.M = num_RIS
        self.N = num_RIS_elements  # per RIS
        self.K = num_users

        self.channel_est_error = channel_est_error

        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        self.bandwidth = bandwidth

        power_t_W = 10 ** (power_t / 10) * 1e-3

        # Initialize the channel matrices
        self.h_km = np.random.rand(self.K, self.M, self.N) + 1j * np.random.rand(self.K, self.M, self.N)
        self.g_km = np.random.rand(self.K, self.M, self.N) + 1j * np.random.rand(self.K, self.M, self.N)

        # Initialize binary association matrix a_km
        self.a_km = np.zeros((self.K, self.M), dtype=int)
        for i in range(self.K):
            chosen_RIS = np.random.choice(self.M)
            self.a_km[i, chosen_RIS] = 1

        # Initialize transmission power rho_k
        self.rho_k_min = 0.0000001  # Minimum allowable transmission power
        self.rho_k = np.random.uniform(self.rho_k_min, power_t_W, size=(self.K,))

        # Initialize phase shift matrix theta_kmn
        imaginary_part = np.random.uniform(0, 2 * np.pi, size=(self.K, self.M, self.N))
        real_part = np.random.choice([-1, 1], size=(self.K, self.M, self.N))
        self.theta_kmn = real_part + 1j * imaginary_part

        # Define the state dimension
        self.state_dim = self.K * self.M  # SNR

        # Define the action dimension
        rho_size = self.K
        theta_size = 2 * self.K * self.M * self.N
        a_size = self.K * self.M  # assuming complex but actually binary
        self.action_dim = rho_size + theta_size + a_size

        print("state_dim in env:", self.state_dim)
        print("action_dim in env:", self.action_dim)


        self.state = None
        self.done = None
        self.episode_t = None

    def reset(self):
        self.episode_t = 0

        # Calculate SNR for each user and RIS
        snr_km = np.zeros((self.K, self.M))
        rho_k_value = self.rho_k
        for k in range(self.K):
            for m in range(self.M):
                a_km_values = self.a_km[k, m]
                h_km_values = self.h_km[k, m, :]
                g_km_values = self.g_km[k, m, :]
                decoded_a_km_values = a_km_values

                rho_k_value = np.squeeze(rho_k_value)

                rho_k_value = np.array(rho_k_value)
            
                snr_km[k, m] = decoded_a_km_values * (
                        rho_k_value[k] * np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values)) ** 2 /
                        self.awgn_var
                )
                #print(f"SNR value: {snr_km[k, m]}")

        # Flatten the SNR matrix
        snr_flat = snr_km.reshape(1, -1)

        #print("snr_flat = ", snr_flat)

        # Construct the state vector
        self.state = np.hstack((snr_flat,))

        return self.state
    
    def step(self, action, power_t):
        self.episode_t += 1

        # Extract rho_k, theta_kmn_real, theta_kmn_imag, and a_km_continuous from action
        rho_k = action[0, :self.K].copy()
        theta_kmn_real = action[0, self.K:self.K + self.K * self.M * self.N].copy()
        theta_kmn_imag = action[0, self.K + self.K * self.M * self.N:2 * self.K * self.M * self.N].copy()
        a_km_continuous = action[0, 2 * self.K * self.M * self.N:].copy()

        # Convert continuous a_km to binary
        a_km_binary = (a_km_continuous > 0).astype(int)

        # Ensure that each row of a_km_binary has exactly one element equal to 1
        a_km = np.zeros((self.K, self.M), dtype=int)
        for i in range(self.K):
            chosen_index = np.argmax(a_km_continuous[i * self.M:(i + 1) * self.M])
            a_km[i, chosen_index] = 1

        # Update the rho_k values
        Pmax = 10 ** (power_t / 10) * 1e-3
        while np.sum(rho_k) > Pmax or (rho_k < 0).any():
            rho_k = np.random.uniform(1e-4, Pmax, size=(self.K,))
        self.rho_k = rho_k

        # Update the theta_kmn values
        self.theta_kmn = theta_kmn_real + 1j * theta_kmn_imag
        self.theta_kmn = self.theta_kmn.reshape(self.K, self.M, self.N)

        # Update a_km
        self.a_km = a_km

        # Calculate SNR and update state
        snr_km = np.zeros((self.K, self.M))
        for k in range(self.K):
            for m in range(self.M):
                if self.a_km[k, m] == 1:
                    snr_km[k, m] = (self.rho_k[k] * np.abs(np.sum(self.theta_kmn[k, m, :] * self.h_km[k, m, :] * self.g_km[k, m, :])) ** 2) / self.awgn_var

        h_kmn_real = np.real(self.h_km).reshape(1, -1)
        h_kmn_imag = np.imag(self.h_km).reshape(1, -1)
        g_kmn_real = np.real(self.g_km).reshape(1, -1)
        g_kmn_imag = np.imag(self.g_km).reshape(1, -1)

        #self.state = np.hstack((snr_km.reshape(1, -1), self.a_km.reshape(1, -1), h_kmn_real, h_kmn_imag, g_kmn_real, g_kmn_imag))
        self.state = np.hstack(snr_km.reshape(1, -1))
        print("self.state = ", self.state)


        reward, opt_reward = self._compute_reward(Pmax)
        done = opt_reward == reward

        return self.state, reward, done, None


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
                snr_km[k, m] = decoded_a_km_values * (rho_k_values*np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values)) ** 2 / sigma)
                #snr_km[k, m] = decoded_a_km_values * (np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values) * rho_k_values) ** 2 / sigma)
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
       

        # Reward with penalty
        # Check if each user's data rate meets the minimum rate requirement
        rate_meets_requirement = (sum_rate[sum_rate > 0] >= r_k_min[sum_rate > 0]).all()


        if rate_meets_requirement:
            c = 0
        else:
            c = self.K
        xi_k = 10 # scale according to sum rate
        reward = sum_rate/1000000 - c*xi_k/1000000
    
        
        # Calculate the reward while considering rate constraints

        opt_reward = reward # example comparison: optimal reward would be without interference
        return reward, opt_reward

    def step(self, action, power_t):
        self.episode_t += 1

        # Separate the action into different components
        rho_k = action[0,:self.K].copy()  # Extract and make a copy of transmission power values
        Pmax = 10 ** (power_t / 10) * 1e-3

        while np.sum(rho_k) > Pmax or (rho_k < 0).any():
            rho_k = np.random.uniform(1e-4, Pmax, size=(self.K,))

        # Update the transmission power values in the action array ?????????????????????????
        action[0, :self.K] = rho_k.reshape(1, self.K)

        #Extract theta_kmn values from action --- check if the same in Actor's forward for correct indices
        theta_kmn_real = action[:, self.K:self.K+self.K*self.M*self.N]
        theta_kmn_imag = action[:, self.K*self.M*self.N:2*self.K*self.M*self.N]

        
        # Update the theta_kmn values in the action array
        action[:, self.K:self.K+self.K*self.M*self.N] = theta_kmn_real.reshape(1, -1)
        action[:, self.K*self.M*self.N:2*self.K*self.M*self.N] = theta_kmn_imag.reshape(1, -1)


        a_km = self.a_km.reshape(1,-1)

        h_kmn_real = np.real(self.h_km).reshape(1,-1)
        h_kmn_imag = np.imag(self.h_km).reshape(1,-1)

        g_kmn_real = np.real(self.g_km).reshape(1,-1)
        g_kmn_imag = np.imag(self.g_km).reshape(1,-1)

        sigma = self.awgn_var  # Example noise variance

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
                snr_km[k, m] = decoded_a_km_values * (rho_k_values[k]*np.abs(np.sum(self.theta_kmn[k, m, :] * h_km_values * g_km_values)) ** 2 / sigma)


        #self.state = np.hstack((snr_km.reshape(1,-1), a_km, h_kmn_real, h_kmn_imag, g_kmn_real, g_kmn_imag))
        self.state = np.hstack(snr_km.reshape(1,-1))       

        #Update env arrays for rho, a_km, theta, h, g ??
        self.rho_k = np.array(action[:, :self.K])

        self.theta_kmn = theta_kmn_real + 1j * theta_kmn_imag
        # Convert it to size (K, M, N)
        theta_kmn_reshaped = self.theta_kmn.reshape(self.K, self.M * self.N).reshape(self.K, self.M, self.N)
        self.theta_kmn = theta_kmn_reshaped
  
        reward, opt_reward = self._compute_reward(Pmax)

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass
