import logging
import math
import os
import os.path as path
from typing import Iterator, Union
import numpy as np

import torch
import torch.optim as optim
from torch import LongTensor
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import src.tb_writter as tb_writter
from src.benchmarks.benchmarks import Benchmark
from src.data import BipartiteGraph, InstanceDataset, TrainSampler
from src.generator import Generator
from src.model import ACMMILP


class Trainer():
    def __init__(self,
                 model: ACMMILP,
                 emb_model: ACMMILP,
                 train_set: InstanceDataset,
                 paths: DictConfig,
                 config: DictConfig,
                 generator_config: DictConfig,
                 benchmark_config: DictConfig,
                 ):
        """
        Trainer for ACMMILP.

        Args:
            model: ACMMILP model
            emb_model: ACMMILP model
            train_set: training dataset
            eval_set: dev dataset
            paths: paths
            config: trainer config
            generator_config: generator config
            benchmark_config: benchmark config
        """
        self.model = model
        self.emb_model = emb_model

        # configurations
        self.config = config
        self.generator_config = generator_config
        self.benchmark_config = benchmark_config

        # laod paths
        self.paths = paths
        self.model_dir = paths.model_dir
        self.samples_dir = paths.samples_dir
        self.dataset_samples_dir = paths.dataset_samples_dir
        self.dataset_stats_dir = paths.dataset_stats_dir
        self.benchmark_dir = paths.benchmark_dir
        self.community_dir = train_set.community_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # load parameters
        self.total_steps = config.steps
        self.step = 0
        self.save_step = config.save_step
        self.update_iters = config.update_iters
        self.update_iters_stage_2 = config.update_iters_stage_2

        self.grad_max_norm = config.grad_max_norm

        # prepare the data loader
        self.batch_size = config.batch_size
        self.batch_repeat_size = config.batch_repeat_size

        self.train_loader: Iterator[BipartiteGraph] = DataLoader(
            dataset=train_set,
            batch_sampler=TrainSampler(
                train_set,
                batch_size=self.batch_size,
                repeat_size=self.batch_repeat_size,
                total_steps=self.total_steps,
            ),
            follow_batch=["x_constraints", "x_variables"],
        )
    
        # init optimizer
        self.get_optimizer(self.model, config)

        # init beta scheduler
        self.beta_cons, self.beta_var = config.beta.cons.min, config.beta.var.min
        self.beta_cons_scheduler = beta_scheduler(config.beta.cons)
        self.beta_var_scheduler = beta_scheduler(config.beta.var)

        self.generator_config = generator_config
        self.benchmark_config = benchmark_config

    def train(self):
        """
        Train the model.
        """
        for data in self.train_loader:
            self.step_train_s1(data)

            if self.step % self.save_step == 0 and self.step >= self.save_start:
                self.step_save()

            if self.step % self.lr_anneal_step == 0:
                self.lr_scheduler.step()

            if self.step / self.total_steps >= 0.5:
                break

        self.save_best_ckpt()

        # stage 2 training
        self.get_optimizer(self.model, self.config, fine_tuning=True)

        # copy best embed model to self.emb_model
        self.load_best_emb_ckpt()

        for data in self.train_loader:
            self.step_train_s2(data)

            if self.step % self.save_step == 0 and self.step >= self.save_start:
                self.step_save(fine_tuning=True)

            if self.step % self.lr_anneal_step == 0:
                self.lr_scheduler.step()

            if self.step >= self.total_steps:
                break

        self.save_best_ckpt(fine_tuning=True)

    def step_train_s1(self, data: Union[Batch, BipartiteGraph]):
        """
        Train the model for one step.
        """
        self.model.train()
        # step_loss_list = []
        for i in range(self.update_iters):
            self.step += 1
            tb_writter.set_step(self.step)
            # random generate batch_size ints, each from 0 to len(data.community_info[current_batch_idx])
            sample_community_idx = [torch.randint(0, len(data.community_info[batch_idx]), (1,)).item()
                                    for batch_idx in range(len(data.community_info))]
            # fetch from data.community_info[current_batch_idx][sample_community_idx]
            # the community info of the current batch
            sample_community_info = [data.community_info[batch_idx][sample_community_idx[batch_idx]]
                                     for batch_idx in range(len(data.community_info))]
            # add data.x_constraints_ptr[batch_idx](int) with sample_community_info[batch_idx](list)
            # to get the indices of the constraints in the current community and convert to tensor
            sample_cons_idx = []
            for batch_idx in range(len(data.community_info)):
                sample_cons_idx.extend(data.x_constraints_ptr[batch_idx].cpu() +
                                       torch.tensor(sample_community_info[batch_idx]))
            sample_cons_idx = LongTensor(sample_cons_idx).cuda()

            data = data.cuda()
            self.model.zero_grad()
            torch.cuda.empty_cache()

            loss = self.model.train_forward(data, sample_cons_idx, beta_cons=self.beta_cons,
                                            beta_var=self.beta_var)
            loss.backward()
            # step_loss_list.append(loss.item())
        # step_loss = sum(step_loss_list) / len(step_loss_list)
            self.beta_cons = self.beta_cons_scheduler.step()
            self.beta_var = self.beta_var_scheduler.step()

            tb_writter.add_scalar("Train/lr", self.lr, self.step)
            tb_writter.add_scalar("Train/beta_cons", self.beta_cons, self.step)
            tb_writter.add_scalar("Train/beta_var", self.beta_var, self.step)
            tb_writter.add_scalar("Train/total_loss", loss, self.step)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_max_norm)
            self.optimizer.step()

            logging.info(
                f"Step {self.step}/{self.total_steps}. Loss: {loss.item():.5f}. beta_cons: {self.beta_cons:.5f}. beta_var: {self.beta_var:.5f}. lr: {self.lr}.")

    def step_train_s2(self, data: Union[Batch, BipartiteGraph]):
        self.model.train()
        self.emb_model.eval()
        # step_loss_list = []
        data = data.cuda()

        community_rank = self.emb_model.sort_community(data, self.update_iters_stage_2)
        for i in range(self.update_iters_stage_2):
            self.step += 1
            tb_writter.set_step(self.step)
            sample_cons_idx = community_rank[i]
            self.model.zero_grad()
            torch.cuda.empty_cache()

            loss = self.model.train_forward(data, sample_cons_idx, beta_cons=self.beta_cons,
                                            beta_var=self.beta_var)
            loss.backward()
            # step_loss_list.append(loss.item())
        # step_loss = sum(step_loss_list) / len(step_loss_list)
            self.beta_cons = self.beta_cons_scheduler.step()
            self.beta_var = self.beta_var_scheduler.step()

            tb_writter.add_scalar("Train/lr", self.lr, self.step)
            tb_writter.add_scalar("Train/beta_cons", self.beta_cons, self.step)
            tb_writter.add_scalar("Train/beta_var", self.beta_var, self.step)
            tb_writter.add_scalar("Train/total_loss", loss, self.step)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_max_norm)
            self.optimizer.step()

            logging.info(
                f"Step {self.step}/{self.total_steps}. Loss: {loss.item():.5f}. beta_cons: {self.beta_cons:.5f}. beta_var: {self.beta_var:.5f}. lr: {self.lr}.")

    def step_save(self, fine_tuning=False):
        """
        Save the model and generate samples.
        """
        self.model.eval()
        self.model.zero_grad()
        if not fine_tuning:
            self.emb_model.load_state_dict(self.model.state_dict())
            self.emb_model.eval()
            self.model.zero_grad()
        torch.cuda.empty_cache()
        if not fine_tuning:
            save_path = path.join(self.model_dir, f"emb_model_step_{self.step}.ckpt")
        else:
            save_path = path.join(self.model_dir, f"model_step_{self.step}.ckpt")
        with torch.no_grad():
            torch.save(self.model.state_dict(), save_path)

        samples_dir = self.samples_dir + f"_step_{self.step}"
        os.makedirs(samples_dir, exist_ok=True)
        generator = Generator(
            model=self.model,
            emb_model=self.emb_model,
            config=self.generator_config,
            templates_dir=self.dataset_samples_dir,
            community_dir=self.community_dir,
            save_dir=samples_dir
        )
        generator.generate()

        benchmark_dir = self.benchmark_dir + f"_step_{self.step}"
        benchmarker = Benchmark(
            config=self.benchmark_config,
            dataset_stats_dir=self.dataset_stats_dir,
        )
        results = benchmarker.assess_samples(
            samples_dir=samples_dir,
            benchmark_dir=benchmark_dir
        )

        info_path = path.join(benchmark_dir, "info.json")
        benchmarker.log_info(
            generator_config=self.generator_config,
            benchmarking_config=self.benchmark_config,
            meta_results=results,
            save_path=info_path,
        )

        tb_writter.add_scalar("Train/similarity",
                              results["distribution"]["score"], self.step)
        tb_writter.add_scalar("Train/solving_time_mean",
                              results["solving"]["solving_time"]["mean"], self.step)
        tb_writter.add_scalar(
            "Train/solving_time_err", results["solving"]["solving_time"]["mean_error"], self.step)
        tb_writter.add_scalar("Train/num_nodes_mean",
                              results["solving"]["num_nodes"]["mean"], self.step)
        tb_writter.add_scalar(
            "Train/num_nodes_err", results["solving"]["num_nodes"]["mean_error"], self.step)

        # we use the sum of mean error as the metric to find the best model
        err = results["solving"]["solving_time"]["mean_error"] + \
            results["solving"]["num_nodes"]["mean_error"]
        self.performance_track.append((self.step, err))

    def save_best_ckpt(self, fine_tuning=False):
        """
        Save the best model.
        """
        self.performance_track.sort(key=lambda x: x[1])
        best_step = self.performance_track[0][0]

        logging.info(f"Best step: {best_step}.")
        if not fine_tuning:
            best_model_path = path.join(
                self.model_dir, f"emb_model_step_{best_step}.ckpt")
        else:
            best_model_path = path.join(
                self.model_dir, f"model_step_{best_step}.ckpt")
        self.model.eval()
        self.model.zero_grad()
        self.model.load_state_dict(torch.load(best_model_path))
        torch.cuda.empty_cache()
        if not fine_tuning:
            save_path = path.join(self.model_dir, f"emb_model_best.ckpt")
        else:
            save_path = path.join(self.model_dir, f"model_best.ckpt")
        with torch.no_grad():
            torch.save(self.model.state_dict(), save_path)
        logging.info(f"Best model saved to {save_path}.")

    def load_best_emb_ckpt(self):
        """
        Load the best embed model.
        """
        best_model_path = path.join(
            self.model_dir, f"emb_model_best.ckpt")
        self.emb_model.eval()
        self.emb_model.zero_grad()
        self.emb_model.load_state_dict(torch.load(best_model_path))
        torch.cuda.empty_cache()

        self.model.eval()
        self.model.zero_grad()
        self.model.load_state_dict(torch.load(best_model_path))
        torch.cuda.empty_cache()

        logging.info(f"Best emb_model loaded from {best_model_path}.")

    @property
    def lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def get_optimizer(self, model, config, fine_tuning=False):
        if fine_tuning:
            self.optimizer = optim.Adam(
                model.parameters(), lr=0.1 * config.lr.init, weight_decay=config.weight_decay)
        else:
            self.optimizer = optim.Adam(
                model.parameters(), lr=config.lr.init, weight_decay=config.weight_decay)

        # init lr scheduler
        self.lr_anneal_step = config.lr.anneal_step
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, config.lr.anneal_rate)

        self.performance_track = []

        self.save_start = config.save_start_stage_2 if fine_tuning else config.save_start_stage_1


class scheduler(object):
    def __init__(self, config) -> None:
        self.beta = config.min
        self.t = 0
        self.warmup = config.warmup
        self.beta_min = config.min
        self.beta_max = config.max
        self.beta_anneal_period = config.anneal_period

    def step(self):
        pass


class linear_schedule(scheduler):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.beta_step = (self.beta_max - self.beta_min) / \
            self.beta_anneal_period

    def step(self):
        if self.t < self.warmup:
            self.beta = self.beta_min
        elif self.t <= self.warmup + self.beta_anneal_period:
            self.beta += self.beta_step
        self.t += 1
        return self.beta


class cyclical_schedule(scheduler):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.beta_num_cycles = config.num_cycles

        self.cycle_period = self.beta_anneal_period // self.beta_num_cycles
        self.linear_period = int(self.cycle_period * 0.5)
        self.beta_step = (self.beta_max - self.beta_min) / self.linear_period

        self.T = max((self.t - self.warmup) // self.cycle_period + 1, 1)
        self.tau = self.t - self.warmup - self.T * self.cycle_period

    def step(self):
        if self.t < self.warmup:
            self.beta = self.beta_min
            self.t += 1
        else:
            if self.tau == 0 and self.T < self.beta_num_cycles:
                self.beta = self.beta_min
                self.T += 1
            elif self.tau <= self.linear_period:
                self.beta = min(self.beta + self.beta_step, self.beta_max)
            self.tau = (self.tau + 1) % self.cycle_period
            self.t += 1

        return self.beta


class sigmoid_schedule(scheduler):
    def __init__(self, config) -> None:
        super(sigmoid_schedule, self).__init__(config)
        self.diff = self.beta_max - self.beta_min
        self.anneal_rate = math.pow(0.01, 1 / self.beta_anneal_period)
        self.weight = 1

    def step(self):
        if self.t < self.warmup:
            self.beta = self.beta_min
        else:
            self.weight = math.pow(self.anneal_rate, self.t - self.warmup)
            self.beta = self.beta_min + self.diff * (1 - self.weight)
        self.t += 1
        return self.beta


class beta_scheduler(object):
    def __init__(self, config) -> None:
        self.mode = config.mode
        if self.mode not in ["linear", "sigmoid", "cyclical"]:
            self.mode = "sigmoid"
        if self.mode == "linear":
            self.schedule = linear_schedule(config)
        elif self.mode == "cyclical":
            self.schedule = cyclical_schedule(config)
        else:
            self.schedule = sigmoid_schedule(config)

    def step(self):
        return self.schedule.step()
