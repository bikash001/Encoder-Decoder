from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
from tensorflow.python.ops import lookup_ops
import sys

UNK_ID = 0
TRAIN = 0
EVAL = 1
INFER = 2

class BatchedInput(
	collections.namedtuple("BatchedInput",
		("initializer", "source", "target_input",
		"target_output", "source_sequence_length",
		"target_sequence_length"))):
	pass


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
	"""Creates vocab tables for src_vocab_file and tgt_vocab_file."""
	src_vocab_table = lookup_ops.index_table_from_file(
		src_vocab_file, default_value=UNK_ID)
	if share_vocab:
		tgt_vocab_table = src_vocab_table
	else:
		tgt_vocab_table = lookup_ops.index_table_from_file(
			tgt_vocab_file, default_value=UNK_ID)
	return src_vocab_table, tgt_vocab_table


def get_iterator(src_dataset,
				tgt_dataset,
				src_vocab_table,
				tgt_vocab_table,
				batch_size,
				sos,
				eos,
				random_seed,
				num_buckets,
				src_max_len=None,
				tgt_max_len=None,
				num_parallel_calls=4,
				output_buffer_size=None,
				skip_count=None,
				num_shards=1,
				shard_index=0,
				reshuffle_each_iteration=True):
 
	if not output_buffer_size:
		output_buffer_size = batch_size * 1000
	src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
	tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
	tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

	src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
	src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
	if skip_count is not None:
		src_tgt_dataset = src_tgt_dataset.skip(skip_count)

	src_tgt_dataset = src_tgt_dataset.shuffle(
		output_buffer_size, random_seed, reshuffle_each_iteration)

	src_tgt_dataset = src_tgt_dataset.map(
		lambda src, tgt: (
			tf.string_split([src]).values, tf.string_split([tgt]).values),
		num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

	# Filter zero length input sequences.
	src_tgt_dataset = src_tgt_dataset.filter(
		lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

	if src_max_len:
		src_tgt_dataset = src_tgt_dataset.map(
			lambda src, tgt: (src[:src_max_len], tgt),
			num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	if tgt_max_len:
		src_tgt_dataset = src_tgt_dataset.map(
			lambda src, tgt: (src, tgt[:tgt_max_len]),
			num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	# Convert the word strings to ids.  Word strings that are not in the
	# vocab get the lookup table's default_value integer.
	src_tgt_dataset = src_tgt_dataset.map(
		lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
			tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
			num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	# Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
	src_tgt_dataset = src_tgt_dataset.map(
		lambda src, tgt: (src,
			tf.concat(([tgt_sos_id], tgt), 0),
			tf.concat((tgt, [tgt_eos_id]), 0)),
		num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
	# Add in sequence lengths.
	src_tgt_dataset = src_tgt_dataset.map(
		lambda src, tgt_in, tgt_out: (
			src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
		num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

	# Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
	def batching_func(x):
		return x.padded_batch(
			batch_size,
			# The first three entries are the source and target line rows;
			# these have unknown-length vectors.  The last two entries are
			# the source and target row sizes; these are scalars.
			padded_shapes=(
				tf.TensorShape([None]),  # src
				tf.TensorShape([None]),  # tgt_input
				tf.TensorShape([None]),  # tgt_output
				tf.TensorShape([]),  # src_len
				tf.TensorShape([])),  # tgt_len
			# Pad the source and target sequences with eos tokens.
			# (Though notice we don't generally need to do this since
			# later on we will be masking out calculations past the true sequence.
			padding_values=(
				src_eos_id,  # src
				tgt_eos_id,  # tgt_input
				tgt_eos_id,  # tgt_output
				0,  # src_len -- unused
				0))  # tgt_len -- unused

	if num_buckets > 1:
		def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
			# Calculate bucket_width by maximum source sequence length.
			# Pairs with length [0, bucket_width) go to bucket 0, length
			# [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
			# over ((num_bucket-1) * bucket_width) words all go into the last bucket.
			if src_max_len:
				bucket_width = (src_max_len + num_buckets - 1) // num_buckets
			else:
				bucket_width = 10

			# Bucket sentence pairs by the length of their source sentence and target
			# sentence.
			bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
			return tf.to_int64(tf.minimum(num_buckets, bucket_id))

		def reduce_func(unused_key, windowed_data):
		  return batching_func(windowed_data)

		batched_dataset = src_tgt_dataset.apply(
			tf.contrib.data.group_by_window(
				key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

	else:
		batched_dataset = batching_func(src_tgt_dataset)
	batched_iter = batched_dataset.make_initializable_iterator()
  
	(src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
		tgt_seq_len) = (batched_iter.get_next())
	return BatchedInput(
		initializer=batched_iter.initializer,
		source=src_ids,
		target_input=tgt_input_ids,
		target_output=tgt_output_ids,
		source_sequence_length=src_seq_len,
		target_sequence_length=tgt_seq_len)


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
	initializer=batched_iter.initializer,
	source=src_ids,
	target_input=None,
	target_output=None,
	source_sequence_length=src_seq_len,
	target_sequence_length=None)


class EncoderDecoder(object):
	"""docstring for EncoderDecoder"""
	def __init__(self, src_file, tgt_file, dev_in, dev_out, src_vocab_file, tgt_vocab_file,
				max_step, random_seed, src_max_len, tgt_max_len,
				src_vocab_size, tgt_vocab_size, embedding_size, max_gradient_norm=5.0,
				sos='<s>', eos='</s>', lr=0.001, batch_size=10, model_dir='../model/', 
				share_vocab=False, time_major=False):
		super(EncoderDecoder, self).__init__()
		self.num_keep_ckpts = 10
		self.model_dir = model_dir
		self.src_file = src_file
		self.tgt_file = tgt_file
		self.dev_in = dev_in
		self.lr = lr
		self.time_major = time_major
		self.sos = sos
		self.eos = eos
		self.dev_out = dev_out
		self.max_step = max_step
		self.embedding_size = embedding_size
		self.max_gradient_norm = max_gradient_norm
		self.random_seed = random_seed
		self.src_max_len = src_max_len
		self.tgt_max_len = tgt_max_len
		self.src_vocab_size = src_vocab_size
		self.src_vocab_file = src_vocab_file
		self.tgt_vocab_file = tgt_vocab_file
		self.is_train = True
		self.num_buckets = 0
		self.batch_size = batch_size
		self.tgt_vocab_size = tgt_vocab_size
		# self.graph = tf.Graph()
		# with self.graph.as_default():
		# 	self.src_vocab_table, self.tgt_vocab_table = create_vocab_tables(
		# 		src_vocab_file, tgt_vocab_file, False)

	def _get_max_time(self, tensor):
		time_axis = 0 if self.time_major else 1
		return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

	def train(self, checkpoints_path):
		train_info = self._build_single_graph(TRAIN)
		print('built train')
		eval_info = self._build_single_graph(EVAL)
		print('built eval')
		infer_info = self._build_single_graph(INFER)
		print('built infer')
		train_graph = train_info['graph']
		eval_graph = eval_info['graph']
		infer_graph = infer_info['graph']

		with train_graph.as_default():
			train_iterator = train_info['iterator']

		with eval_graph.as_default():
			eval_iterator = eval_info['iterator']
			
		with infer_graph.as_default():
			infer_iterator = infer_info['iterator']
			
		checkpoints_path = "/tmp/model/checkpoints"

		train_sess = tf.Session(graph=train_graph)
		eval_sess = tf.Session(graph=eval_graph)
		infer_sess = tf.Session(graph=infer_graph)

		with train_info['graph'].as_default():
			logits = train_info['logits']
			self.global_step = tf.Variable(0, trainable=False)
			crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=train_info['iterator'].target_output, logits=logits)

			max_time = self._get_max_time(train_info["iterator"].target_output)

			target_weights = tf.sequence_mask(
				train_info["iterator"].target_sequence_length, max_time, dtype=logits.dtype)
			train_loss = (tf.reduce_sum(crossent * target_weights) /
				self.batch_size)

			# Calculate and clip gradients
			params = tf.trainable_variables()
			gradients = tf.gradients(train_loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(
				gradients, self.max_gradient_norm)

			# Optimization
			optimizer = tf.train.AdamOptimizer(self.lr)
			update_step = optimizer.apply_gradients(
				zip(clipped_gradients, params), global_step=self.global_step)
			
			initializer = tf.global_variables_initializer()
			tables_initializer = tf.tables_initializer()
			train_info['loss'] = train_loss
			train_info['step'] = update_step

		with eval_info['graph'].as_default():
			logits = eval_info['logits']
			crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=eval_info["iterator"].target_output, logits=logits)

			max_time = self._get_max_time(eval_info["iterator"].target_output)

			target_weights = tf.sequence_mask(
				eval_info["iterator"].target_sequence_length, max_time, dtype=logits.dtype)
			loss = (tf.reduce_sum(crossent * target_weights) /
				self.batch_size)
			eval_initializer = tf.global_variables_initializer()
			eval_tab_init = tf.tables_initializer()
			eval_info['loss'] = loss

		train_sess.run(initializer)
		train_sess.run(tables_initializer)
		train_sess.run(train_iterator.initializer)
		eval_sess.run(eval_initializer)
		eval_sess.run(eval_tab_init)
		print('first initialized')
		# Summary writer
		summary_writer = tf.summary.FileWriter(self.model_dir+'/model', train_info['graph'])

		# with tf.Session(graph=train_info['graph']) as sess:
		# 	step = 0
		# 	while step < self.max_step:
		# 		sess.run(update_step)
		# 		step += 1
		EVAL_STEPS = 10
		for i in xrange(self.max_step):
			print('----------------------------> %d' %i)
			train_sess.run(update_step)
			if i % EVAL_STEPS == 0:
				checkpoint_path = train_info['saver'].save(train_sess, checkpoints_path, global_step=i)
				eval_info['saver'].restore(eval_sess, checkpoint_path)
				eval_sess.run(eval_iterator.initializer, feed_dict = {
									eval_info['src_file_placeholder']: self.dev_in,
									eval_info['tgt_file_placeholder']: self.dev_out
								})
				while True:
					try:
						loss = eval_sess.run(eval_info['loss'])
					except tf.errors.OutOfRangeError:
						print(loss)
						break
			# if i % INFER_STEPS == 0:
			# 	checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
			# 	infer_model.saver.restore(infer_sess, checkpoint_path)
			# 	infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
			# 	while data_to_infer:
			# 		infer_model.infer(infer_sess)


	# mode = 0 (train), 1 (eval), 2 (infer)
	def _build_single_graph(self, mode):
		graph = tf.Graph()
		info = {'graph': graph}
		with graph.as_default():
			self.src_vocab_table, self.tgt_vocab_table = create_vocab_tables(
				self.src_vocab_file, self.tgt_vocab_file, False)
			if mode == TRAIN:
				src_dataset = tf.data.TextLineDataset(self.src_file)
				tgt_dataset = tf.data.TextLineDataset(self.tgt_file)
				itr = get_iterator(src_dataset,
						tgt_dataset,
						self.src_vocab_table,
						self.tgt_vocab_table,
						self.batch_size,
						self.sos,
						self.eos,
						self.random_seed,
						self.num_buckets,
						src_max_len=self.src_max_len,
						tgt_max_len=self.tgt_max_len)
				info['iterator'] = itr

			elif mode == EVAL:
				src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
				tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
				src_dataset = tf.data.TextLineDataset(src_file_placeholder)
				tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
				itr = get_iterator(src_dataset,
						tgt_dataset,
						self.src_vocab_table,
						self.tgt_vocab_table,
						self.batch_size,
						self.sos,
						self.eos,
						self.random_seed,
						self.num_buckets,
						src_max_len=self.src_max_len,
						tgt_max_len=self.tgt_max_len)
				info['src_file_placeholder'] = src_file_placeholder
				info['tgt_file_placeholder'] = tgt_file_placeholder
				info['iterator'] = itr

			elif mode == INFER:
				src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
				batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
				src_dataset = tf.data.Dataset.from_tensor_slices(
					src_placeholder)
				itr = get_infer_iterator(
					src_dataset,
					self.src_vocab_table,
					batch_size=batch_size_placeholder,
					eos=self.eos,
					src_max_len=None)
				info['src_placeholder'] = src_placeholder
				info['batch_size_placeholder'] = batch_size_placeholder
				info['iterator'] = itr

			else:
				raise ValueError('invalid mode')

			encoder_inputs = itr.source
			decoder_inputs = itr.target_input
			target_outputs = itr.target_output
			source_sequence_length = itr.source_sequence_length
			tgt_sequence_length = itr.target_sequence_length

			# Embedding
			embedding_encoder = tf.get_variable(
				"embedding_encoder", [self.src_vocab_size, self.embedding_size]) 
			embedding_decoder = tf.get_variable(
				"embedding_decoder", [self.tgt_vocab_size, self.embedding_size])

			# Look up embedding:
			#   encoder_inputs: [max_time, batch_size]
			#   encoder_emb_inp: [max_time, batch_size, embedding_size]
			encoder_emb_inp = tf.nn.embedding_lookup(
				embedding_encoder, encoder_inputs)
			
			# Build RNN cell
			encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)
			# Run Dynamic RNN
			#   encoder_outputs: [max_time, batch_size, num_units]
			#   encoder_state: [batch_size, num_units]
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
			 	encoder_cell, encoder_emb_inp, dtype=tf.float32,
			 	sequence_length=source_sequence_length, time_major=False)

			# maximum_iteration: The maximum decoding steps.
			maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
			
			# DECODER
			# Build RNN cell
			decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size)

			if mode != INFER:
				decoder_emb_inp = tf.nn.embedding_lookup(
					embedding_decoder, decoder_inputs)
				
				# Helper
				helper =  tf.contrib.seq2seq.TrainingHelper(
					decoder_emb_inp, tgt_sequence_length, time_major=False)

				# Projection
				projection_layer = tf.layers.Dense(
					  self.tgt_vocab_size, use_bias=False, name="output_projection")

				# Decoder
				decoder = tf.contrib.seq2seq.BasicDecoder(
				  decoder_cell, helper, encoder_state)

				# Dynamic decoding
				outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
				sample_id = outputs.sample_id
				# print outputs
				logits = projection_layer(outputs.rnn_output)
				saver = tf.train.Saver(
					tf.global_variables(), max_to_keep=self.num_keep_ckpts)
				info['logits'] = logits
				info['sample_id'] = sample_id
				info['saver'] = saver
			
			else:
				tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.sos)), tf.int32)
				tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.eos)), tf.int32)
				# Projection
				projection_layer = tf.layers.Dense(
					  self.tgt_vocab_size, use_bias=False, name="output_projection")

				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
					embedding_decoder, tf.fill([self.batch_size], tgt_sos_id), tgt_eos_id)
				# Decoder
				decoder = tf.contrib.seq2seq.BasicDecoder(
					decoder_cell, helper, encoder_state,
					output_layer=projection_layer)
				# Dynamic decoding
				outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(
					decoder, maximum_iterations=maximum_iterations)
				sample_id = outputs.sample_id
				saver = tf.train.Saver(
					tf.global_variables(), max_to_keep=self.num_keep_ckpts)
				info['sample_id'] = sample_id
				info['saver'] = saver			
		return info

	def fit(self):
		graph = tf.Graph()
		with graph.as_default():
			self.logits, _ = self._build_graph(TRAIN)
			self.global_step = tf.Variable(0, trainable=False)
			crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=itr.target_output, logits=self.logits)

			max_time = self._get_max_time(itr.target_output)

			target_weights = tf.sequence_mask(
				tgt_sequence_length, max_time, dtype=self.logits.dtype)
			train_loss = (tf.reduce_sum(crossent * target_weights) /
				self.batch_size)

			# Calculate and clip gradients
			params = tf.trainable_variables()
			gradients = tf.gradients(train_loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(
				gradients, max_gradient_norm)

			# Optimization
			optimizer = tf.train.AdamOptimizer(self.lr)
			self.update_step = optimizer.apply_gradients(
				zip(clipped_gradients, params), global_step=self.global_step)
		
		with tf.Session(graph=graph) as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.tables_initializer())
			sess.run(itr.initializer)
			
			step = 0
			while step < self.max_step:
				sess.run(update_step)
				step += 1
				
	def inference(self, src, tgt):
		pass
		# if type(src) == type(tgt) == str:
		# 	with self.graph.as_default():

		# self.is_train = False
		# with tf.Session(graph=self.graph) as sess:
		# 	sess.run(translation)

if __name__ == '__main__':
	arg = sys.argv[1:]
	arg = arg[:6] + list(map(int, arg[6:]))
	model = EncoderDecoder(*arg)
	model.train('../model/')