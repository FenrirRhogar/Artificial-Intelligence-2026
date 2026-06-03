This project contains the basic structure for completing the first phase of the Second Project assigned during the AI course in TUC, during the 2025-2026 academic year. 
Three distinct AI agents (Hill Climbing, Random Restart Hill Climbing, and Simulated Annealing) agents should be implemented. All the agents will be tested across different domains in the [ns gym](https://nsgym.io/).

## Installation Instructions
For the installation of the project, it is suggested to use a virtual environment (such as [conda](https://www.anaconda.com/docs/getting-started/miniconda/main)). Python 3.10 should be running in this environment.

To download the code, please use 
```bash
git clone https://github.com/leoBakop/AI-agents-in-ns-gym.git
```
Then, you only have to run (after activating the new environment)
```python
pip3 install ns-gym
```

To test the installation, please run 
```python
python3 test/test_ns_gymnasium.py
```

Assuming a correct installation, you should be able to see a rendering of the environment, and some printed actions (a list of random integers) on your terminal.