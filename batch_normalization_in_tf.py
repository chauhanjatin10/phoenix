import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs),name="X")

is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

bn_params =  {
	'is_training': is_training,
	'decay': 0.99,
	'update_collections': None
}

with tf.contrib.framework.arg_scope([fully_connected],
		normalizer_fn=batch_norm,normalizer_params=bn_params):
	hidden1 = fully_connected(X,n_hidden1)
	hidden2 = fully_connected(hidden1,n_hidden2)
	logits = fully_connected(hidden2,n_outputs,activation_fn=None)


with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=y,logits=logits)
	loss = tf.reduce_mean(xentropy,name="loss")

learning_rate = 0.02

with tf.name_scope("train"):
	optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
			momentum=0.8)
	training_operation = optimizer.minimize(loss)

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(output,y,1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()

n_epochs = 100

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		for X_batch,y_batch in zip(X_batches,y_batches):
			sess.run(training_operation,
				feed_dict={is_training:True, X: X_batch, y_batch: y_batch})
		accuracy_score = accuracy.eval(
			feed_dict={is_training: False, X: X_test_scaled, y: y_test})
		print(accuracy_score)