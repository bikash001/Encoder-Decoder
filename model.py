from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
from tensorflow.python.ops import lookup_ops
import sys

UNK_ID = 0


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


def get_iterator(src_file,
				tgt_file,
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
 
  src_dataset = tf.data.TextLineDataset(src_file)
  tgt_dataset = tf.data.TextLineDataset(tgt_file)
    
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

class EncoderDecoder(object):
	"""docstring for EncoderDecoder"""
	def __init__(self, src_file, tgt_file, src_vocab_file, tgt_vocab_file, 
				share_vocab=False, time_major=False):
		super(EncoderDecoder, self).__init__()
		self.src_file 
		self.src_vocab_table, self.tgt_vocab_table = create_vocab_tables(
			src_vocab_file, tgt_vocab_file, False)

	def _get_max_time(self, tensor):
		time_axis = 0 if self.time_major else 1
		return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

	def _build_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			itr = get_iterator(self.src_file,
								self.tgt_file,
								self.src_vocab_table,
								self.tgt_vocab_table,
								self.batch_size,
								self.sos,
								self.eos,
								self.random_seed,
								self.num_buckets,
								src_max_len=self.src_max_len,
								tgt_max_len=self.tgt_max_len)


		encoder_inputs = itr.source
		decoder_inputs = itr.target_input
		target_outputs = itr.target_output
		source_sequence_length = itr.source_sequence_length
		tgt_sequence_length = itr.target_sequence_length
  # decoder_inputs = tf.placeholder(tf.int32, shape=[max_time, batch_size],  name='decoder_inputs')
  # encoder_inputs = tf.constant(np.array([[1,2,3], [0,1,2]]))
  # encoder_inputs = tf.placeholder(tf.int32, shape=[max_time, batch_size], name='encoder_inputs')
  # source_sequence_length = [3,3]
  # source_sequence_length = tf.placeholder(tf.int32, name='source_sequence_length')
  # tgt_sequence_length = tf.placeholder(tf.int32, shape=(1,), name='tgt_sequence_length')

  # Embedding
  embedding_encoder = tf.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size]) 
  embedding_decoder = tf.get_variable(
    "embedding_decoder", [tgt_vocab_size, embedding_size])

  # Look up embedding:
  #   encoder_inputs: [max_time, batch_size]
  #   encoder_emb_inp: [max_time, batch_size, embedding_size]
  encoder_emb_inp = tf.nn.embedding_lookup(
    embedding_encoder, encoder_inputs)
  decoder_emb_inp = tf.nn.embedding_lookup(
    embedding_decoder, decoder_inputs)

  # seqs = tf.constant([[0,2,3,4,0], [1,2,3,4,5]])
  # embedded_seqs = tf.contrib.layers.embed_sequence(seqs, vocab_size=6, embed_dim=3)
  # return embedded_seqs

  # Build RNN cell
  encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)

  # Run Dynamic RNN
  #   encoder_outputs: [max_time, batch_size, num_units]
  #   encoder_state: [batch_size, num_units]
  encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_cell, encoder_emb_inp, dtype=tf.float32,
      sequence_length=source_sequence_length, time_major=False)

  # DECODER
  # Build RNN cell
  decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)

  # Helper
  helper = tf.contrib.seq2seq.TrainingHelper(
      decoder_emb_inp, tgt_sequence_length, time_major=False)
  
  # Projection
  projection_layer = tf.layers.Dense(
          tgt_vocab_size, use_bias=False, name="output_projection")

  # Decoder
  decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_cell, helper, encoder_state)
  
  # Dynamic decoding
  outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
  # print outputs
  logits = projection_layer(outputs.rnn_output)

  ## Loss
  if mode == "TRAIN":

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_outputs, logits=logits)
    
    max_time = get_max_time(target_outputs)
    
    target_weights = tf.sequence_mask(
        tgt_sequence_length, max_time, dtype=logits.dtype)
    train_loss = (tf.reduce_sum(crossent * target_weights) /
        batch_size)

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, max_gradient_norm)

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(
        zip(clipped_gradients, params))
    
    return update_step, train_loss, logits, target_outputs
  else:
    loss = None

  return logits


