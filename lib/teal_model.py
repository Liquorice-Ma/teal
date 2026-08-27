import pickle
import time
import json
import sys
import os
from copy import deepcopy
from tqdm import tqdm
from networkx.readwrite import json_graph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .teal_actor import TealActor
from .teal_env import TealEnv
from .utils import print_


class Teal():
    def __init__(self, teal_env, teal_actor, lr, early_stop,
                 eval_admm_steps=0):
        """Initialize Teal model.

        Args:
            teal_env: teal environment
            num_layer: number of flowGNN layers
            lr: learning rate
            early_stop: whether to early stop
            eval_admm_steps: repair budget used consistently by validation
                and testing when selecting trained checkpoints
        """

        self.env = teal_env
        self.actor = teal_actor

        # init optimizer
        self.base_lr = lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        # cosine decay to lr/10: the MLU reward needs a large initial lr
        # to make progress, but oscillates without annealing
        self.actor_scheduler = None

        # early stop when val result no longer changes
        self.early_stop = early_stop
        # Model selection must evaluate the same end-to-end pipeline used at
        # test time. Otherwise checkpoints are ranked without repair even
        # when the deployed metric includes repair.
        self.eval_admm_steps = eval_admm_steps
        # best validation objective seen so far (for model checkpointing)
        self.best_val = None
        self.best_state = None
        # always available: restart screening validates regardless of
        # whether early stopping is enabled
        self.val_reward = []

    def train(self, num_epoch, batch_size, num_sample,
              num_restart=1, warmup_epoch=5):
        """Train Teal model.

        Args:
            num_epoch: number of training epoch
            batch_size: batch size
            num_sample: number of samples in COMA reward
            num_restart: number of candidate initializations to screen
            warmup_epoch: epochs per candidate during screening
        """

        if num_restart > 1:
            self._select_init(
                num_restart, warmup_epoch, batch_size, num_sample)
        self._run_epochs(num_epoch, batch_size, num_sample, track_best=True)
        if self.best_state is not None:
            print('best val_obj {:.6f}'.format(self.best_val), flush=True)
            self.actor.load_state_dict(self.best_state)
        self.actor.save_model()

    def _select_init(self, num_restart, warmup_epoch, batch_size, num_sample):
        """Briefly train several random inits and keep the best one.

        A sizable fraction of random initializations never escape a bad
        region on this problem (validation objectives of 2.4-3.2 versus a
        normal 1.5), which makes individual runs unusable. Screening is done
        purely on the validation slice, so the test slice stays untouched.
        """

        best_state, best_val = None, None
        for restart in range(num_restart):
            if restart > 0:
                self.actor.reinit_parameters()
            self._reset_optimizer()
            self.val_reward = []
            self._run_epochs(
                warmup_epoch, batch_size, num_sample, track_best=False,
                desc='Warmup {}'.format(restart))
            self.val()
            val = self.val_reward[-1]
            print('restart {} warmup val_obj {:.6f}'.format(restart, val),
                  flush=True)
            if best_val is None or self._better(val, best_val):
                best_val, best_state = val, deepcopy(self.actor.state_dict())
        print('selected init, warmup val_obj {:.6f}'.format(best_val),
              flush=True)
        self.actor.load_state_dict(best_state)
        self._reset_optimizer()
        self.val_reward = []
        self.best_val, self.best_state = None, None

    def _reset_optimizer(self):
        """Fresh optimizer and scheduler (used between restarts)."""

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.base_lr)
        self.actor_scheduler = None

    def _run_epochs(self, num_epoch, batch_size, num_sample,
                    track_best=True, desc='Training epoch'):
        """Run num_epoch training epochs."""

        for epoch in range(num_epoch):

            if self.actor_scheduler is None:
                self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.actor_optimizer, T_max=max(num_epoch, 1),
                    eta_min=self.actor_optimizer.param_groups[0]['lr'] * 0.1)

            self.env.reset('train')
            # reload graph in case the demand set was rebuilt (demand_split)
            self.actor.FlowGNN.refresh_graph()

            ids = range(self.env.idx_start, self.env.idx_stop)
            loop_obj = tqdm(
                [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)],
                desc=f"{desc} {epoch}/{num_epoch}: ")

            for idx in loop_obj:
                loss = 0
                for _ in idx:
                    torch.cuda.empty_cache()

                    # get observation
                    obs = self.env.get_obs()
                    # get action
                    raw_action, log_probability = self.actor.evaluate(obs)
                    # get reward
                    reward, info = self.env.step(
                        raw_action, num_sample=num_sample)
                    loss += -(log_probability*reward).mean()

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                # break
            self.actor_scheduler.step()

            # early stop
            if track_best and self.early_stop:
                self.val()
                print('epoch {} val_obj {:.6f}'.format(
                    epoch, self.val_reward[-1]), flush=True)
                # keep the best-validation weights: the objective oscillates
                # under the high learning rates needed for the MLU reward,
                # so the final epoch is not necessarily the best model
                if self._is_best(self.val_reward[-1]):
                    self.best_val = self.val_reward[-1]
                    self.best_state = deepcopy(self.actor.state_dict())
                if len(self.val_reward) > 20 and abs(
                        sum(self.val_reward[-20:-10])/10
                        - sum(self.val_reward[-10:])/10) < 0.0001:
                    break

    def _better(self, val, reference):
        """Return whether val beats reference for the current objective."""

        return val < reference \
            if self.env.obj == 'min_max_link_util' else val > reference

    def _is_best(self, val):
        """Return whether val improves on the best seen validation result.

        Lower is better for min_max_link_util, higher for total_flow.
        """

        if self.best_val is None:
            return True
        return val < self.best_val \
            if self.env.obj == 'min_max_link_util' else val > self.best_val

    def val(self):
        """Validating Teal model."""

        self.actor.eval()
        self.env.reset('val')
        self.actor.FlowGNN.refresh_graph()

        rewards = 0
        for idx in range(self.env.idx_start, self.env.idx_stop):

            # get observation
            problem_dict = self.env.render()
            obs = self.env.get_obs()
            # get action
            raw_action = self.actor.act(obs)
            # Evaluate the same end-to-end protocol used during testing,
            # including the configured repair budget.
            reward, info = self.env.step(
                raw_action, num_admm_step=self.eval_admm_steps)
            # show satisfied demand instead of total flow
            rewards += reward.item()/problem_dict['total_demand']\
                if self.env.obj == 'total_flow' else reward.item()
        self.val_reward.append(
            rewards/(self.env.idx_stop - self.env.idx_start))

    def test(self, num_admm_step, output_header, output_csv, output_dir):
        """Test Teal model.

        Args:
            num_admm_step: number of ADMM steps
            output_header: header of the output csv
            output_csv: name of the output csv
            output_dir: directory to save output solution
        """

        self.actor.eval()
        self.env.reset('test')
        # reload graph: with demand_split the test-time demand set is
        # rebuilt from test TMs while model weights stay unchanged
        self.actor.FlowGNN.refresh_graph()

        with open(output_csv, "a") as results:
            print_(",".join(output_header), file=results)

            runtime_list, obj_list = [], []
            loop_obj = tqdm(
                range(self.env.idx_start, self.env.idx_stop),
                desc="Testing: ")

            for idx in loop_obj:

                # get observation
                problem_dict = self.env.render()
                obs = self.env.get_obs()
                # get action
                start_time = time.time()
                raw_action = self.actor.act(obs)
                runtime = time.time() - start_time
                # get reward
                reward, info = self.env.step(
                    raw_action, num_admm_step=num_admm_step)
                # add runtime in transforming, ADMM, rounding
                runtime += info['runtime']
                runtime_list.append(runtime)
                # show satisfied demand instead of total flow
                obj_list.append(
                    reward.item()/problem_dict['total_demand']
                    if self.env.obj == 'total_flow' else reward.item())

                # display avg runtime, obj
                loop_obj.set_postfix({
                    'runtime': '%.4f' % (sum(runtime_list)/len(runtime_list)),
                    'obj': '%.4f' % (sum(obj_list)/len(obj_list)),
                    })

                # save solution matrix
                sol_mat = info['sol_mat']
                torch.save(sol_mat, os.path.join(
                    output_dir,
                    "{}-{}-{}-teal_objective-{}_{}-paths_"
                    "edge-disjoint-{}_dist-metric-{}_sol-mat.pt".format(
                        problem_dict['problem_name'],
                        problem_dict['traffic_model'],
                        problem_dict['traffic_seed'],
                        problem_dict['obj'],
                        problem_dict['num_path'],
                        problem_dict['edge_disjoint'],
                        problem_dict['dist_metric'])))

                PLACEHOLDER = ",".join("{}" for _ in output_header)
                result_line = PLACEHOLDER.format(
                    problem_dict['problem_name'],
                    problem_dict['num_node'],
                    problem_dict['num_edge'],
                    problem_dict['traffic_seed'],
                    problem_dict['scale_factor'],
                    problem_dict['traffic_model'],
                    problem_dict['total_demand'],
                    "Teal",
                    problem_dict['num_path'],
                    problem_dict['edge_disjoint'],
                    problem_dict['dist_metric'],
                    problem_dict['obj'],
                    reward,
                    runtime)
                print_(result_line, file=results)
                # break
