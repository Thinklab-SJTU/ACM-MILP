import os.path as path

import hydra
import torch
from omegaconf import DictConfig

from src import ACMMILP, Benchmark, Generator, set_cpu_num, set_seed


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def generate(config: DictConfig):
    """
    Generate instances using ACMMILP.
    """
    set_seed(config.seed)
    set_cpu_num(config.num_workers + 1)
    torch.cuda.set_device(config.cuda)

    model_path = path.join(config.paths.model_dir, "model_best.ckpt")
    model = ACMMILP.load_model(config, model_path)

    emb_model_path = path.join(config.paths.model_dir, "emb_model_best.ckpt")
    emb_model = ACMMILP.load_model(config, emb_model_path)

    generator = Generator(
        model=model,
        emb_model=emb_model,
        config=config.generator,
        templates_dir=config.paths.dataset_samples_dir,
        community_dir=config.paths.community_info_dir,
        save_dir=config.paths.samples_dir,
    )
    generator.generate()

    benchmarker = Benchmark(
        config=config.benchmarking,
        dataset_stats_dir=config.paths.dataset_stats_dir,
    )
    results = benchmarker.assess_samples(
        samples_dir=config.paths.samples_dir,
        benchmark_dir=config.paths.benchmark_dir,
    )

    info_path = path.join(config.paths.benchmark_dir, "info.json")
    benchmarker.log_info(
        generator_config=config.generator,
        benchmarking_config=config.benchmarking,
        meta_results=results,
        save_path=info_path,
    )


if __name__ == '__main__':
    generate()
