# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import argparse
import re
import time
import tensorflow as tf

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

tf.compat.v1.disable_v2_behavior()

parser = argparse.ArgumentParser(description="Monodepth TensorFlow implementation.")

parser.add_argument("--mode", type=str, help="train or test", default="train")
parser.add_argument("--model_name", type=str, help="model name", default="monodepth")
parser.add_argument(
    "--encoder", type=str, help="type of encoder, vgg or resnet50", default="vgg"
)
parser.add_argument(
    "--dataset",
    type=str,
    help="dataset to train on, kitti, or cityscapes",
    default="kitti",
)
parser.add_argument("--data_path", type=str, help="path to the data", required=True)
parser.add_argument(
    "--filenames_file", type=str, help="path to the filenames text file", required=True
)
parser.add_argument("--input_height", type=int, help="input height", default=256)
parser.add_argument("--input_width", type=int, help="input width", default=512)
parser.add_argument("--batch_size", type=int, help="batch size", default=8)
parser.add_argument("--num_epochs", type=int, help="number of epochs", default=50)
parser.add_argument(
    "--learning_rate", type=float, help="initial learning rate", default=1e-4
)
parser.add_argument(
    "--lr_loss_weight", type=float, help="left-right consistency weight", default=1.0
)
parser.add_argument(
    "--alpha_image_loss",
    type=float,
    help="weight between SSIM and L1 in the image loss",
    default=0.85,
)
parser.add_argument(
    "--disp_gradient_loss_weight",
    type=float,
    help="disparity smoothness weigth",
    default=0.1,
)
parser.add_argument(
    "--do_stereo", help="if set, will train the stereo model", action="store_true"
)
parser.add_argument(
    "--wrap_mode",
    type=str,
    help="bilinear sampler wrap mode, edge or border",
    default="border",
)
parser.add_argument(
    "--use_deconv", help="if set, will use transposed convolutions", action="store_true"
)
parser.add_argument(
    "--num_gpus", type=int, help="number of GPUs to use for training", default=1
)
parser.add_argument(
    "--num_threads",
    type=int,
    help="number of threads to use for data loading",
    default=8,
)
parser.add_argument(
    "--output_directory",
    type=str,
    help="output directory for test disparities, if empty outputs to checkpoint folder",
    default="",
)
parser.add_argument(
    "--log_directory",
    type=str,
    help="directory to save checkpoints and summaries",
    default="",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="path to a specific checkpoint to load",
    default="",
)
parser.add_argument(
    "--retrain",
    help="if used with checkpoint_path, will restart training from step zero",
    action="store_true",
)
parser.add_argument(
    "--full_summary",
    help="if set, will keep more data for each summary. Warning: the file can become very large",
    action="store_true",
)

args = parser.parse_args()


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def count_text_lines(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return len(lines)


def train(params):
    """Training loop."""

    num_training_samples = count_text_lines(args.filenames_file)
    steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
    num_total_steps = params.num_epochs * steps_per_epoch

    boundaries = [int((3 / 5) * num_total_steps), int((4 / 5) * num_total_steps)]
    values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
    learning_rate = tf.compat.v1.train.piecewise_constant(
        tf.compat.v1.train.get_or_create_global_step(), boundaries, values
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    print(f"Total number of samples: {num_training_samples}")
    print(f"Total number of steps: {num_total_steps}")

    dataloader = MonodepthDataloader(
        args.data_path, args.filenames_file, params, args.dataset, args.mode
    )
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch

    # split for each gpu
    left_splits = tf.split(left, args.num_gpus, 0)
    right_splits = tf.split(right, args.num_gpus, 0)

    tower_grads = []
    tower_losses = []
    tower_image_losses = []
    tower_dispgrad_losses = []
    tower_lr_losses = []

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
        for i in range(args.num_gpus):
            with tf.device(f"/gpu:{i}"):
                with tf.compat.v1.variable_scope(
                    f"model_{i}", reuse=tf.compat.v1.AUTO_REUSE
                ):
                    model = MonodepthModel(
                        params, args.mode, left_splits[i], right_splits[i]
                    )
                    loss = model.total_loss
                    tower_image_losses.append(model.image_loss)
                    tower_dispgrad_losses.append(model.disp_gradient_loss)
                    tower_lr_losses.append(model.lr_loss)
                    tower_losses.append(loss)

                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(
        grads, global_step=tf.compat.v1.train.get_global_step()
    )

    total_loss = tf.math.reduce_mean(tower_losses)
    dispgrad_loss = tf.math.reduce_mean(tower_dispgrad_losses)
    lr_loss = tf.math.reduce_mean(tower_lr_losses)
    image_loss = tf.math.reduce_mean(tower_image_losses)

    tf.compat.v1.summary.scalar("learning_rate", learning_rate)
    tf.compat.v1.summary.scalar("total_loss", total_loss)
    summary_op = tf.compat.v1.summary.merge_all()

    # SESSION
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess = tf.compat.v1.Session(config=config)

    # SAVER
    summary_writer = tf.compat.v1.summary.FileWriter(
        args.log_directory + "/" + args.model_name, sess.graph
    )
    train_saver = tf.compat.v1.train.Saver()

    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print(f"Number of trainable parameters: {total_num_parameters}")

    # INIT
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    coordinator = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coordinator)

    # LOAD CHECKPOINT IF SET
    if args.checkpoint_path != "":
        train_saver.restore(sess, args.checkpoint_path.split(".")[0])

        if args.retrain:
            sess.run(tf.compat.v1.train.get_global_step().assign(0))

    # GO!
    start_step = sess.run(tf.compat.v1.train.get_global_step())
    start_time = time.time()
    for step in range(start_step, num_total_steps):
        before_op_time = time.time()
        _, loss_value, image_loss, lr_loss, dispgrad_loss = sess.run(
            [apply_gradient_op, total_loss, image_loss, lr_loss, dispgrad_loss]
        )
        wandb.log(
            {
                "image_loss": image_loss.eval(session=sess),
                "disp_gradient_loss": dispgrad_loss.eval(session=sess),
                "lr_loss": lr_loss.eval(session=sess),
                "total_loss": total_loss.eval(session=sess),
            }
        )
        duration = time.time() - before_op_time
        if step and step % 100 == 0:
            examples_per_sec = params.batch_size / duration
            time_sofar = (time.time() - start_time) / 3600
            training_time_left = (num_total_steps / step - 1.0) * time_sofar
            print_string = "batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h"
            print(
                print_string.format(
                    step, examples_per_sec, loss_value, time_sofar, training_time_left
                )
            )
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=step)
        if step and step % 10000 == 0:
            train_saver.save(
                sess,
                args.log_directory + "/" + args.model_name + "/model",
                global_step=step,
            )

    train_saver.save(
        sess,
        args.log_directory + "/" + args.model_name + "/model",
        global_step=num_total_steps,
    )


def test(params):
    """Test function."""
    dataloader = MonodepthDataloader(
        args.data_path, args.filenames_file, params, args.dataset, args.mode
    )
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = MonodepthModel(params, args.mode, left, right)

    # SESSION
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess = tf.compat.v1.Session(config=config)

    # SAVER
    train_saver = tf.compat.v1.train.Saver()

    # INIT
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    coordinator = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == "":
        restore_path = tf.compat.v1.train.latest_checkpoint(
            args.log_directory + "/" + args.model_name
        )
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print(f"Now testing {num_test_samples} files.")
    disparities = np.zeros(
        (num_test_samples, params.height, params.width), dtype=np.float32
    )
    disparities_pp = np.zeros(
        (num_test_samples, params.height, params.width), dtype=np.float32
    )
    for step in range(num_test_samples):
        disp = sess.run(model.disp_left_est[0])
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())

    print("Done.")

    output_directory = (
        args.output_directory
        if args.output_directory
        else os.path.dirname(args.checkpoint_path)
    )
    np.save(os.path.join(output_directory, "disparities.npy"), disparities)
    np.save(os.path.join(output_directory, "disparities_pp.npy"), disparities_pp)

    print("Done.")


def main():
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        lr=args.learning_rate,
        model_name=args.model_name,
    )
    wandb.init(
        project=params.model_name,
        config={
            "num_epochs": params.num_epochs,
            "learning_rate": params.lr,
        },
    )

    if args.mode == "train":
        train(params)
    elif args.mode == "test":
        test(params)


if __name__ == "__main__":
    main()
