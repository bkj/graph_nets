#!/usr/bin/env python

"""
    shortest_path.py
    
    See
      https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/shortest_path.ipynb
    for some more explanation and discussion
    
    !! Graph generation takes a nontrivial amount of time
"""


import json
import argparse
import numpy as np
from time import time

import tensorflow as tf

from graph_nets import utils_tf
from graph_nets.demos import models

from helpers import (
  create_placeholders,
  create_loss_ops,
  create_feed_dict,
  make_all_runnable_in_session,
  compute_accuracy,
)

# --
# Helpers

def create_loss_ops(target_op, output_ops):
  return [
      tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
      tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
      for output_op in output_ops
  ]

# --
# CLI
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--iters', type=int, default=10000)
    
    parser.add_argument('--batch-size-train', type=int, default=32)
    parser.add_argument('--batch-size-test', type=int, default=100)
    
    parser.add_argument('--steps-train', type=int, default=10)
    parser.add_argument('--steps-test', type=int, default=10)
    
    parser.add_argument('--node-range-train', type=str, default='8,17')
    parser.add_argument('--node-range-test', type=str, default='16,33')
    
    parser.add_argument('--eval-interval', type=int, default=32)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--theta', type=int, default=20)
    
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    args.node_range_train = eval(args.node_range_train)
    args.node_range_test  = eval(args.node_range_test)
    
    return args

if __name__ == '__main__':
  args = parse_args()
  
  np.random.seed(args.seed)
  tf.set_random_seed(args.seed + 111)
  tf.reset_default_graph()
  rand = np.random.RandomState(seed=args.seed + 222)
  
  input_ph, target_ph = create_placeholders(
    rand=rand, 
    batch_size=args.batch_size_train,
    num_nodes_min_max=args.node_range_train,
    theta=args.theta
  )
  
  model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
  train_output_ops = model(input_ph, args.steps_train)
  test_output_ops  = model(input_ph, args.steps_test)
  
  # train loss
  train_loss_ops = create_loss_ops(target_ph, train_output_ops)
  train_loss_op  = sum(train_loss_ops) / args.steps_train
  
  # test loss
  test_loss_ops = create_loss_ops(target_ph, test_output_ops)
  test_loss_op  = test_loss_ops[-1]  # Loss from final processing step.
  
  # optimizer
  optimizer = tf.train.AdamOptimizer(args.lr)
  step_op   = optimizer.minimize(train_loss_op)
  
  input_ph  = utils_tf.make_runnable_in_session(input_ph)
  target_ph = utils_tf.make_runnable_in_session(target_ph)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # --
  # Run
  
  history = []
  
  start_time = time()
  for it in range(args.iters):
    
    feed_dict, _ = create_feed_dict(
      rand=rand,
      batch_size=args.batch_size_train,
      num_nodes_min_max=args.node_range_train, 
      theta=args.theta,
      input_ph=input_ph,
      target_ph=target_ph,
    )
    
    train_values = sess.run({
        "step"    : step_op,
        "target"  : target_ph,
        "loss"    : train_loss_op,
        "outputs" : train_output_ops
    }, feed_dict=feed_dict)
    
    if not it % args.eval_interval:
      
      feed_dict, raw_graphs = create_feed_dict(
          rand=rand,
          batch_size=args.batch_size_test,
          num_nodes_min_max=args.node_range_test,
          theta=args.theta,
          input_ph=input_ph,
          target_ph=target_ph,
        )
      
      test_values = sess.run({
          "target"  : target_ph,
          "loss"    : test_loss_op,
          "outputs" : test_output_ops
      }, feed_dict=feed_dict)
      
      train_correct, train_solveed = compute_accuracy(
          train_values["target"], train_values["outputs"][-1], use_edges=True)
      
      test_correct, test_solved = compute_accuracy(
          test_values["target"], test_values["outputs"][-1], use_edges=True)
      
      history.append({
        "iteration"     : it,
        "elapsed"       : time() - start_time,
        "train_loss"    : train_values['loss'],
        "train_correct" : train_correct,
        "train_solved"  : train_solveed,
        "test_loss"     : test_values["loss"],
        "test_correct"  : test_correct,
        "test_solved"   : test_solved,
      })
      print(json.dumps(history[-1]))




