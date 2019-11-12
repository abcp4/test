from src.models import architecture_manager as am
import tensorflow as tf

model_name = 'miniception_C'
model_description = """inception network architecture"""

def make_model(n_classes, alpha=1, seed=12345, n_blocks=4, width=99, height=99, n_channels=1):
    #network configuration
    x, y, lr_placeholder = am.create_input( width, height, n_channels )
    
    #xavier initialization
    initializer = tf.contrib.layers.xavier_initializer(seed = seed)
    input = x
    for n_block in range(n_blocks):
        block_conv, block_res = am.blockC(input, initializer, num_maps1=2**(n_block)*alpha, num_maps2=2**(n_block+1)*alpha )
        print('shape: ', block_conv.shape)

        pool = tf.layers.max_pooling2d(inputs=block_conv, pool_size=[5, 5], strides=2, padding='same')
        print( 'pooling: ', pool.shape )
        input = pool
    
    flatten = tf.contrib.layers.flatten(pool)
    
    fc1 = tf.contrib.layers.fully_connected(flatten, num_outputs=alpha*32, 
    							activation_fn=tf.nn.relu)
    
    output_logits = tf.contrib.layers.fully_connected(fc1, num_outputs=n_classes, 
    							activation_fn=None)
    
    with tf.variable_scope('Prediction'):
        cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
    
    return x, y, lr_placeholder, output_logits, cls_prediction, model_description, None

def make_model_loss(y,lr_placeholder, output_logits):
    with tf.variable_scope('Train'):
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
        tf.summary.scalar('loss', loss)

        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_placeholder, name='Adam-op').minimize(loss)

        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(output_logits, 1), y, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        
        return loss, accuracy, optimizer
