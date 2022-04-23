def train(self, args):
    #--------- YOUR CODE HERE --------------
    
    
    lr = 0.001#args.learning_rate
    optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
    
    loss_fcn = nn.MSELoss()
    N = 10000
    replay_buffer = ReplayBuffer(N)


    # target_network.load_state_dict(main_network.state_dict())
    # self.q_network.eval()

    gamma = 0.999
    num_episodes = 800
    batch_size = 64
    epsilon = 0.2
    num_steps = 200
    update_steps = 5
    reward_collection = []

    for i_episode in range(num_episodes):
        
        state  = self.env.reset()
        # state = self.env.arm.get_state()
        episode_reward = 0
        print("episode, ", i_episode)

        for mini_step in range(num_steps):
            # print("batch, ", mini_step)
            # With probability ε, at = random
            if np.random.random() < epsilon:
                discrete_action = np.random.randint(0, 7)

            # otherwise at = maxaQA(s, a)
            else:
                # observation = states + goal --> [q1, q2, qdot1, qdot2, xgoal, ygoal]
                state_cur = self.env.arm.get_state().flatten()
                goal = self.env.goal
                obs = np.concatenate((state_cur, goal), axis=None, dtype=np.float32)

                # print("obs, ", obs)
                discrete_action = self.q_network.select_discrete_action(torch.from_numpy(obs), self.device)

            # discrete to continuous
            continuous_action = self.q_network.action_discrete_to_continuous(discrete_action)
            
            # Execute action
            next_state, reward, done, _ = self.env.step(continuous_action)
            
            # update reward 
            episode_reward += reward
            
            reward_collection.append(episode_reward)

            replay_buffer.put((state, discrete_action, reward, next_state, done)) # not sure if action should be disc or cont?
            
            state = next_state

            # train network
            if mini_step > batch_size:

                state_, action_, reward_, next_state_, done_ = replay_buffer.sample(batch_size)

                target_max = torch.max(self.t_network.forward(next_state_, self.device), dim=1)[0]
                updated = torch.FloatTensor(reward_).to(self.device) + gamma * target_max * (1 - torch.FloatTensor(done_).to(self.device))

                expected = self.q_network.forward(state_, self.device).gather(1, torch.FloatTensor(action_).type(torch.int64).unsqueeze(1)).squeeze()
                loss = loss_fcn(updated, expected)
                
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if done:
                    
                    break

            # if state is terminal 
            if next_state is None:
                break
    
            if mini_step % update_steps == 0:
                self.t_network.load_state_dict(self.q_network.state_dict())

        print(f"episode={i_episode}, global_step: {mini_step}, episode_reward: {episode_reward} ")
        self.save_model(i_episode, episode_reward, args)