import spacy
from collections import Counter
import numpy as np
import random
import tensorflow as tf

##############################变量和超参数设定##############################

pos_file = "../datasets/comment_classifer/pos.txt"
neg_file = "../datasets/comment_classifer/neg.txt"
REGULARIZATION_RATE = 0.05
LEARNING_RATE = 0.01
# 隐藏层结构，数组长度为隐藏层层数，各元素数字代表该层的节点数
HIDDEN_NODE_COUNT = [500,500]
BATCH_SIZE = 500
TRAINING_STEPS = 1000
MODEL_DIR = "../models/comment_classifer/"

##############################数据预处理###################################

# 读取文件并分词
def process_file(file):
    lex = []
    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            sentence = line.strip()
            words = nlp(sentence.lower())
            lex += words
        return lex

# 创建词库
def create_lexicon(pos_file, neg_file):
    # 获取正负评论文件所有词语
    lex_pos = process_file(pos_file)
    lex_neg = process_file(neg_file)
    lex_unlemma = lex_pos + lex_neg
    # 词干化
    lex_lemma = [word.lemma_ for word in lex_unlemma]
    word_count = Counter(lex_lemma)
    # 去无用常用词，同时去重
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
            lex.append(word)
    return lex

# 评论向量化（lex:词库；review:评论；clf:评论对应的分类）
def string_to_vector(lex, review, clf):
    # 每条评论分词+词干化处理
    sentence = review.strip()
    words = nlp(sentence.lower())
    words = [word.lemma_ for word in words]
    # 对每条评论生成一个与词库大小相同的稀疏向量，对每个词语找词频，产生一个基于词频的向量
    features = [0.0] * len(lex)
    for word in words:
        if word in lex:
            features[lex.index(word)] += 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
    return [features, clf]

# 文件及数据处理，转化成向量，生成载入神经网络系统的数据集
def normalize_dataset(lex):
    dataset = []
    # 打开文件并向量化（分类：[0,1]代表负面评论，[1,0]代表正面评论）
    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [[ 0.,  1.,  0., ...,  0.,  0.,  0.], [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])  # [[ 0.,  0.,  0., ...,  0.,  0.,  0.], [0,1]]
            dataset.append(one_sample)
    return np.array(dataset)

##############################模型训练#####################################

# DNN结构
# 传入特征向量集，设置的隐藏层个数，正则化设定(训练时传入，测试时为None)
def DNN_neural_network(x, y_, hidden_node_count, regularizer, test_or_not):
    reuse = test_or_not
    # 每次循环即一层隐藏层，通过tf.variable_scope隔离各层
    num = 0
    for i in hidden_node_count:
        with tf.variable_scope("layer_%d"%(num+1), reuse=reuse):
            biases = tf.get_variable("bias", shape=[i], initializer=tf.constant_initializer(0.0))
            if num == 0:
                weights = tf.get_variable("weights", shape=[x.shape[1], i],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                if regularizer != None:
                    tf.add_to_collection("losses", regularizer(weights))
                layer = tf.nn.relu(tf.add(tf.matmul(x, weights), biases))
            else:
                weights = tf.get_variable("weights", shape=[hidden_node_count[num-1], i],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                if regularizer != None:
                    tf.add_to_collection("losses", regularizer(weights))
                layer = tf.nn.relu(tf.add(tf.matmul(layer, weights), biases))
            num += 1
    # 定义输出层
    with tf.variable_scope("layer_out", reuse=reuse):
        weights = tf.get_variable("weights", shape=[hidden_node_count[-1], y_.shape[1]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(weights))
        biases = tf.get_variable("bias", shape=[y_.shape[1]], initializer=tf.constant_initializer(0.0))
        layer_out = tf.add(tf.matmul(layer, weights), biases)
    return layer_out

# DNN神经网络模型训练
def train_DNN(X, Y, model_save_dir):
    # 定义feed的x,y_
    x = tf.placeholder("float", shape=[None, X.shape[1]], name="x_input")
    y_ = tf.placeholder(tf.int64, shape=[None, Y.shape[1]], name="y_input")
    # 定义正则项
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 定义预测分类结果的过程(训练过程中传入正则项)
    y = DNN_neural_network(x, y_, HIDDEN_NODE_COUNT, regularizer, False)
    # 定义全局步数
    global_step = tf.Variable(0, trainable=False)
    # 定义损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    loss = tf.reduce_mean(cross_entropy)
    # 优化函数
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=loss, global_step=global_step)
    # 初始化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 模型训练
        for i in range(TRAINING_STEPS):
            # 每轮迭代都取出batch_size尺寸的数据
            random_position = random.randint(0, X.shape[0] - BATCH_SIZE)
            x_batch = X[random_position: random_position + BATCH_SIZE, :]
            y_batch = Y[random_position: random_position + BATCH_SIZE, :]
            _, loss_value, step = sess.run([optimizer, loss, global_step],
                                           feed_dict={x: x_batch, y_: y_batch})
            if step % 50 == 0:
                print("loss_value:"+str(loss_value)+",step:"+str(step))
    # 保存模型
        saver.save(sess, model_save_dir + "model.ckpt", global_step=global_step)

# DNN神经网络模型测试
def eval_DNN(X, Y, model_save_dir):
    # 定义feed的x,y_
    x = tf.placeholder("float", shape=[None, X.shape[1]], name="x_input")
    y_ = tf.placeholder(tf.int64, shape=[None, Y.shape[1]], name="y_input")
    # 定义预测分类结果的过程(测试过程中不传入正则项)
    y = DNN_neural_network(x, y_, HIDDEN_NODE_COUNT, None, True)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        accuracy_score = sess.run(accuracy,feed_dict={x: X, y_: Y})
        print("accuracy_score:" + str(accuracy_score))

##############################主程序######################################

if __name__ == '__main__':
    # 引入分词模型
    nlp = spacy.load('en')
    # 创建词库
    lex = create_lexicon(pos_file, neg_file)
    # 生成神经网络输入的数据集
    dataset = normalize_dataset(lex)
    # 打乱数据集序列的顺序
    random.shuffle(dataset)
    # 划分训练测试集(7:3)
    train_size = int(len(dataset) * 0.7)
    train_data = dataset[0: train_size]
    test_data = dataset[train_size:]
    # 划分训练测试的特征和标签
    train_x = np.array(list(train_data[:, 0]))
    train_y = np.array(list(train_data[:, 1]))
    test_x = np.array(list(test_data[:, 0]))
    test_y = np.array(list(test_data[:, 1]))
    # 训练模型
    train_DNN(train_x, train_y, MODEL_DIR)
    # 模型测试
    eval_DNN(test_x, test_y, MODEL_DIR)