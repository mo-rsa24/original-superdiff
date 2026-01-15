from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

import run_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval_fid", "eval_fid_stoch", "eval_joint_fid", "eval_joint_fid_stoch", "fid_stats"], "Running mode: train, eval or fid_stats")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("chkpts", None,
                    "paths to checkpoints for joint evaluation (comma separated)")
flags.DEFINE_boolean("use_wandb", False, "Whether to use Weights & Biases logging.")
flags.DEFINE_string("wandb_id", None, "WandB ID for resuming runs.")
flags.DEFINE_string("wandb_project", os.environ.get("WANDB_PROJECT", ""), "Weights & Biases project")
flags.DEFINE_string("wandb_name",    os.environ.get("WANDB_NAME",    ""), "W&B run name")
flags.DEFINE_string("wandb_tags",    os.environ.get("WANDB_TAGS",    ""), "Comma-separated W&B tags")
flags.DEFINE_string("wandb_group",   os.environ.get("WANDB_RUN_GROUP",""), "W&B group")
flags.DEFINE_bool("use_wandb", bool(int(os.environ.get("WANDB", "0"))), "Enable W&B logging")
flags.DEFINE_string("git_branch", os.environ.get("GIT_BRANCH",""), "Git branch (for logging)")
flags.DEFINE_string("git_commit_short", os.environ.get("GIT_COMMIT_SHORT",""), "Short git commit (for logging)")
flags.mark_flags_as_required(["workdir", "config", "mode"])

def launch(argv):
  tf.config.experimental.set_visible_devices([], "GPU")
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  if FLAGS.mode == "train":
    run_lib.train(FLAGS.config, FLAGS.workdir, use_wandb=FLAGS.use_wandb, wandb_id=FLAGS.wandb_id)
  elif FLAGS.mode == "eval_fid":
    run_lib.evaluate_fid(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, stoch=False)
  elif FLAGS.mode == "eval_fid_stoch":
    run_lib.evaluate_fid(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, stoch=True)
  elif FLAGS.mode == "eval_joint_fid":
    checkpoints = list(map(lambda _s: _s.strip(), FLAGS.chkpts.split(',')))
    run_lib.evaluate_joint_fid(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, checkpoints, stoch=False)
  elif FLAGS.mode == "eval_joint_fid_stoch":
    checkpoints = list(map(lambda _s: _s.strip(), FLAGS.chkpts.split(',')))
    run_lib.evaluate_joint_fid(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, checkpoints, stoch=True)
  elif FLAGS.mode == "fid_stats":
    run_lib.fid_stats(FLAGS.config, FLAGS.workdir)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(launch)
