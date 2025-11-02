# mjslm

This repository now includes a simple example of training a tabular Q-learning agent using OpenAI Gym.

## Train a Gym agent

Run the training script with default parameters (FrozenLake-v1 environment):

```bash
python train_gym_agent.py
```

Customise the run by selecting a different environment or tweaking hyperparameters:

```bash
python train_gym_agent.py --env Taxi-v3 --episodes 10000 --learning-rate 0.5 --discount 0.9
```

To persist the learned Q-table or save training statistics:

```bash
python train_gym_agent.py --save-path outputs/q_table.npy --summary-path outputs/training_summary.json
```

The script prints a JSON summary to the terminal containing the mean reward, the best reward, a moving average of the last 100 episodes, the average episode length, the proportion of episodes that achieved a positive reward, and a short natural-language interpretation tying the metrics together.  You can open the optional summary file in any JSON viewer or plotting tool to inspect how the reward curve and success rate evolve over time.

After training, you may evaluate the greedy policy derived from the learned Q-table and optionally render the environment:

```bash
python train_gym_agent.py --eval-episodes 20 --render-eval
```

The evaluation metrics (number of test episodes, mean reward, reward history, average episode length, success rate, and a plain-language explanation) are appended to the printed JSON block and saved inside the summary file when ``--summary-path`` is provided.  These values let you answer questions such as “How often does the trained agent now reach the goal?” or “How long does it usually take?”

### Play with or against the trained agent

Once a training run completes you can immediately jump into an interactive session.  The script always reuses the single Q-table it just learned, so every automated player you see is acting from the exact same policy that produced the evaluation metrics.

Watch the agent act greedily with rendering and a short pause between steps:

```bash
python train_gym_agent.py --episodes 5000 --play-mode agent --play-render --play-episodes 3 --play-pause 0.75
```

Take manual control yourself by selecting discrete actions in the terminal:

```bash
python train_gym_agent.py --env Taxi-v3 --episodes 8000 --play-mode human --play-render
```

If you would like to compare your instincts against the policy, use the ``compare`` mode, which shows the agent’s recommended action before you choose:

```bash
python train_gym_agent.py --play-mode compare --play-render
```

Each interactive session is summarised under the ``interactive_play`` entry in the printed JSON so you can review how many steps you took, whether the goals were reached, and how the outcomes stacked up against the automated runs.

### What the results mean

* **Mean reward** – higher is better.  On sparse-reward environments like FrozenLake a mean reward close to `1.0` indicates the agent reliably reaches the goal.
* **Success rate** – the percentage of episodes with a positive reward.  This is often easier to reason about than the raw reward, especially for games with binary win/lose outcomes.
* **Mean episode length** – how many steps the agent needed on average.  If this drops over time, the policy is becoming more direct.
* **Interpretation field** – a generated sentence summarising the above, useful for a quick “story” of how training or evaluation went.

Plotting the moving average or success rate against episode number (for example by loading the JSON into a notebook) will let you visually gauge whether the policy is still improving or has converged.

### Moving toward your goals

1. **Match your personal play style** – Create or adapt a Gym environment that faithfully models the game you care about (for instance Hong Kong Mahjong).  Capture states and rewards that encode the decisions you would make, then use the provided script as a starting point for data collection and policy evaluation.  Recording examples of your own play lets you compare the learned policy’s success rate or action choices against your real decisions.
2. **Prototype a Small Language Model (SLM) helper** – Once the environment exposes the right state/action space, you can use the saved Q-table as a lightweight policy lookup on mobile devices.  For richer reasoning or natural-language hints, combine the tabular policy with a compact neural or language model that translates board states into strategy tips.  The JSON summaries produced here are ideal telemetry for validating that the on-device helper still behaves optimally after you port the policy.

Install dependencies if needed:

```bash
pip install gym numpy
# or, if you prefer the actively maintained fork
pip install gymnasium numpy
```

## Train the Hong Kong Mahjong self-play policy

The `train_hk_mahjong_selfplay.py` script runs four identical policies against
each other inside the full Hong Kong Mahjong environment and logs how the East
seat improves over time. To get started:

1. Install the required libraries (Gymnasium and NumPy are mandatory; install
   Matplotlib if you want the learning-curve plot or omit it and run with
   ``--no-plot``):

   ```bash
   pip install gymnasium numpy matplotlib
   ```

2. Launch training. This example runs 300 rounds and saves the learning curve
   under `outputs/hk_mahjong_learning_curve.png`:

   ```bash
   python train_hk_mahjong_selfplay.py --episodes 300
   ```

   The script prints periodic summaries such as the scaled score delta for the
   East seat (dealer), its moving-average reward, and the rolling win rate. At
   the end of the run you will see a message indicating where the PNG plot was
   written.

3. Explore additional flags when you are ready to customise the run:

   ```bash
   python train_hk_mahjong_selfplay.py \
       --episodes 500 \
       --reward-scale 120 \
       --temperature 0.25 \
       --output-dir results \
       --plot-name my_curve.png
   ```

   Lower `--reward-scale` values make the agent more conservative about risky
   discards (avoiding 出衝), while higher values encourage pressing for extra
   fan before declaring a win (叫糊). Use `--log-every 1` to see a summary line
   for each round and combine it with `--verbose` to print the full Mahjong log
   for every hand if you need to audit decisions.
