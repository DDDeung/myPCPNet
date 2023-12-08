import wandb, hydra, omegaconf
from transformers import set_seed
from train import PCPNetTrainer
import utils

@hydra.main(config_path=".", config_name="config.yaml", version_base='1.1')
def main(args):
    # wandb.config = omegaconf.OmegaConf.to_container(
    #     args, resolve=True, throw_on_missing=True
    # )
    wb_logger = wandb.init(project='PCPNet', name=args.experiment_name)
    set_seed(args.train.seed)
    # wb_logger.config.update(args)

    trainer = PCPNetTrainer(args=args,wb_logger=wb_logger)
    trainer.train()
    trainer.save()

if __name__ == '__main__':
    main()