import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup
import time
from defrcn.dataloader import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import logging


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from defrcn.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


class MultiStreamTrainer(Trainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        cfg_exp = cfg.clone()
        cfg_exp.defrost()
        cfg_exp.DATASETS.TRAIN = cfg_exp.DATASETS.EXP_TRAIN
        cfg_exp.SOLVER.IMS_PER_BATCH = int(cfg_exp.SOLVER.IMS_PER_BATCH)/2
        cfg_exp.freeze()

        cfg_imp = cfg.clone()
        cfg_imp.defrost()
        cfg_imp.DATASETS.TRAIN = cfg_imp.DATASETS.IMP_TRAIN
        cfg_imp.SOLVER.IMS_PER_BATCH = int(cfg_imp.SOLVER.IMS_PER_BATCH) / 2
        cfg_imp.freeze()

        data_loader_exp = build_detection_train_loader(cfg_exp)
        self.data_loader_exp = data_loader_exp
        self._data_loader_exp_iter = iter(data_loader_exp)

        data_loader_imp = build_detection_train_loader(cfg_imp)
        self.data_loader_imp = data_loader_imp
        self._data_loader_imp_iter = iter(data_loader_imp)

        logger = logging.getLogger(__name__)
        logger.info("Using multi stream trainer......")

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_exp = next(self._data_loader_exp_iter)
        data_imp = next(self._data_loader_imp_iter)
        data.extend(data_exp)
        data.extend(data_imp)

        data_time = time.perf_counter() - start
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = MultiStreamTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
