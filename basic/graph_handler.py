import gzip
import json
from json import encoder
import os

import tensorflow as tf

from basic.evaluator import Evaluation, F1Evaluation
from my.utils import short_floats
from metrics.evaluate_off import evaluate
import pickle


class GraphHandler(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.writer = None
        self.save_path = os.path.join(config.save_dir, config.model_name)
        self.best_squad_f1 = 0

    def initialize(self, sess):
        sess.run(tf.initialize_all_variables())
        if self.config.load:
            self._load(sess)

        if self.config.mode == 'train':
            self.writer = tf.train.SummaryWriter(self.config.log_dir, graph=tf.get_default_graph())
        if self.config.mode == 'test' and self.config.save_on_best_f1:
            self._load(sess,latest_filename="checkpoint_best")

    def save(self, sess, global_step=None):
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        saver.save(sess, self.save_path, global_step=global_step)

    def _load(self, sess,latest_filename=None):
        config = self.config
        vars_ = {var.name.split(":")[0]: var for var in tf.all_variables()}
        if config.load_ema:
            ema = self.model.var_ema
            for var in tf.trainable_variables():
                del vars_[var.name.split(":")[0]]
                vars_[ema.average_name(var)] = var
        saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)

        if config.load_path:
            save_path = config.load_path
        elif config.load_step > 0:
            save_path = os.path.join(config.save_dir, "{}-{}".format(config.model_name, config.load_step))
        else:
            save_dir = config.save_dir
            if latest_filename is not None:
                checkpoint = tf.train.get_checkpoint_state(save_dir,latest_filename=latest_filename)
            else:
                checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
        print("Loading saved model from {}".format(save_path))
        saver.restore(sess, save_path)

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def dump_eval(self, e, precision=2, path=None):
        assert isinstance(e, Evaluation)
        if self.config.dump_pickle:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.pklz".format(e.data_type, str(e.global_step).zfill(6)))
            with gzip.open(path, 'wb', compresslevel=3) as fh:
                pickle.dump(e.dict, fh)
        else:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
            with open(path, 'w') as fh:
                json.dump(short_floats(e.dict, precision), fh)

    def dump_answer(self, e, global_step=0, sess=None,path=None):
        assert isinstance(e, Evaluation)
        path = path or os.path.join(self.config.answer_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
        with open(path, 'w') as fh:
            json.dump(e.id2answer_dict, fh)
        if self.config.save_on_best_f1:
            e,f = evaluate(os.path.join(self.config.source_dir,self.config.dev_file_name),path)
            if f > self.best_squad_f1:
                self.best_squad_f1 = f
                saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
                saver.save(sess, self.save_path, global_step=global_step, latest_filename='checkpoint_best')


