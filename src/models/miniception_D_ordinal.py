from src.models import architecture_manager as am
import tensorflow as tf
import numpy as np

model_name = 'miniception_D_ordinal'
model_description = """inception network architecture"""

def make_model( n_classes, alpha=1, seed=12345, n_blocks=4, width=99, height=99, n_channels=1):
    #network configuration
    x, y, lr_placeholder = am.create_input( width, height, n_channels )

    #xavier initialization
    initializer = tf.contrib.layers.xavier_initializer(seed = seed)

    # limiares do p-rank
    tb = tf.Variable(np.zeros(n_classes).astype(np.float32))

    input = x
    for n_block in range(n_blocks):
        block_conv, block_res = am.blockD_2(input, initializer, num_maps1=2**(n_block)*alpha, num_maps2=2**(n_block+1)*alpha)
        print('shape: ', block_conv.shape)

        pool = tf.layers.max_pooling2d(inputs=block_conv, pool_size=[5, 5], strides=2, padding='same')
        print( 'pooling: ', pool.shape )
        input = pool

    #flatten = tf.contrib.layers.flatten(pool)
    flatten = tf.reduce_mean( pool, axis = [1, 2] )
    fc1 = tf.contrib.layers.fully_connected(flatten, num_outputs=alpha*32, weights_initializer = initializer,
                                activation_fn=tf.nn.relu)

    # score de ativacao [n_batches x 1]
    output_logits = tf.contrib.layers.fully_connected(fc1, num_outputs=1, weights_initializer = initializer,
                                activation_fn=None)

    # score por classe
    t_scores = tf.cumsum(output_logits - tb,axis=1)

    with tf.variable_scope('Prediction'):
        cls_prediction = tf.argmax(t_scores, axis=1, name='predictions')
    
    return x, y, lr_placeholder, output_logits, cls_prediction, model_description, t_scores

def make_model_loss(y, lr_placeholder, t_scores) :

    t_pred = tf.argmax(t_scores, axis=1)
    # loss do p-rank
    t_indices = tf.to_int64( tf.range( tf.shape( t_scores )[0] ) )
    t_scores_p = tf.gather_nd( t_scores,
                               tf.stack(
                                   (t_indices, tf.to_int64( t_pred )), -1 )
                               )
    t_scores_y = tf.gather_nd( t_scores,
                               tf.stack(
                                   (t_indices, y), -1 )
                               )

    with tf.variable_scope( 'Train' ) :
        with tf.variable_scope( 'Loss' ) :
            loss = tf.reduce_sum(t_scores_p) - tf.reduce_sum(t_scores_y)
        tf.summary.scalar( 'loss', loss )

        with tf.variable_scope( 'Optimizer' ) :
            optimizer = tf.train.AdamOptimizer( learning_rate = lr_placeholder, name = 'Adam-op' ).minimize( loss )

        with tf.variable_scope( 'Accuracy' ) :
            correct_prediction = tf.equal( t_pred, y, name = 'correct_pred' )
            accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ), name = 'accuracy' )
        tf.summary.scalar( 'accuracy', accuracy )

        return loss, accuracy, optimizer
