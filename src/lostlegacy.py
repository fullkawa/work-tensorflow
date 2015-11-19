# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

"""
ゲームの定義

本サンプルでは"LostLegacy"から最小限の要素を抽出したゲームを定義する。
@see http://one-draw.jp/lostlegacy/top.html
"""

"""
コンポーネント定義
  key: コンポーネント名
  value: 数量
"""
components = {
  "C1":1, "C2":1, "C3":1, "C4":2
}

"""
フィールド定義
  key: フィールド名
  value: フィールドサイズ(最大)
"""
fields = {
  "deck":5,
  "dig":1,
  "player_hand":1, "player_draw":1, "player_trash":5,
  "next_pl_hand":1, "next_pl_draw":1, "next_pl_trash":5
}

def explode_def(dic):
  ary = []
  for key, value in dic.items():
    for i in range(value):
      ary.append('%s[%d]' % (key, i))
  return sorted(ary)

acomponents = explode_def(components)
print "components:", acomponents
components_num = len(acomponents)

afields = explode_def(fields)
print "fields:", afields
fields_num = len(afields)

dsize = components_num * fields_num
print

# 

def inference(ph_play, ph_check):
  """
  with tf.name_scope('inference') as scope:
    Wp = tf.Variable(tf.zeros([dsize, 1]), name="weight:play")
    Wc = tf.Variable(tf.zeros([dsize, 1]), name="weight:check")
    
    y = tf.nn.softmax(tf.matmul(ph_play, Wp) + tf.matmul(ph_check, Wc))
  """
  Wp = tf.Variable(tf.zeros([dsize, 1]))
  Wc = tf.Variable(tf.zeros([dsize, 1]))
  
  y = tf.nn.softmax(tf.matmul(ph_play, Wp) + tf.matmul(ph_check, Wc))
  return y

def loss(output, ph_supervisor_labels):
  with tf.name_scope('loss') as scope:
    cross_entropy = -tf.reduce_sum(ph_supervisor_labels * tf.log(output))
    tf.scalar_summary("entropy", cross_entropy)
  return cross_entropy

def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

df11 = pd.read_csv('../data/step1-1.csv', header=None)
df12 = pd.read_csv('../data/step1-2.csv', header=None)
df13 = pd.read_csv('../data/step1-3.csv', header=None)

ary11 = np.array(df11.values).reshape(1, dsize)
ary12 = np.array(df12.values).reshape(1, dsize)
ary13 = np.array(df13.values).reshape(1, dsize)

with tf.Graph().as_default():
  ph_play_input = tf.placeholder("float", [None, dsize], name="placeholder_play")
  ph_check_input = tf.placeholder("float", [None, dsize], name="placeholder_check")
  ph_supervisor_labels = tf.placeholder("float", [None, dsize], name="placeholder_supervisor")
  feeds = {
    ph_play_input: ary11,
    ph_check_input: ary12,
    ph_supervisor_labels: ary13
  }
  
  output = inference(ph_play_input, ph_check_input)
  loss = loss(output, ph_supervisor_labels)
  training_op = training(loss)
  summary_op = tf.merge_all_summaries()
  
  init = tf.initialize_all_variables()
  
  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)
    
    for step in range(1000):
      sess.run(training_op, feed_dict=feeds)
      if step % 100 == 0:
        print sess.run(loss, feed_dict=feeds)
        summary_str = sess.run(summary_op, feed_dict=feeds)
        summary_writer.add_summary(summary_str, step)
