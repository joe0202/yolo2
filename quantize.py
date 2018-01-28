import tensorflow as tf
def edit_graph_fn(graph, ops_name, map_op_ts):#对位于opsname中的Input,若其位于map_op_ts
    for node in graph.get_operations():
        if node.name in ops_name:
            for i in range(len(node.inputs)):
                old_key = node.inputs[i].name
                if old_key in map_op_ts:
                    ours = graph.get_operation_by_name(map_op_ts[old_key].op.name)
                    target = graph.get_operation_by_name(node.name)
                    #print("append: {} \n -->\n {}\n".format(ours.name, node.name))
                    target._update_input(i, ours.outputs[0])//对输入tensor加上mask,并将其作为ops_name节点的新输入


def find_anchors_and_deps(graph,
                          anchor_white_list=['Conv2D', 'BiasAdd', 'MatMul', 'Relu', 'AvgPool', 'MaxPool', 'Concat', 'Reshape', 'Squeeze'],
                          anchor_name_filter_white = ['lstm_cell'],
                          anchor_name_filter_black = ['projection', 'gradients', 'Adam'],
                          deps_black_list=['Placeholder']):
    assert 'Add' not in anchor_white_list, 'Please remove Add'
    assert 'Mul' not in anchor_white_list, 'Please remove Mul'
    anchors = set()
    deps = set()
    
    for current_node in graph.get_operations():
        if current_node.type in anchor_white_list:

            cond = True
            '''
            for namefilter in anchor_name_filter_white:
                if (namefilter in current_node.name):
                    cond = True
                    break
            for namefilter in anchor_name_filter_black:
                if (namefilter in current_node.name):
                    cond = False
                    break
            '''
            if cond:
                flag = False
                for input_node in current_node.inputs:
                    if input_node.op.type not in deps_black_list and input_node.dtype not in ['int32']:
                        #print(input_node.name)
                        deps.add(input_node)
                        flag = True
                if flag:
                    anchors.add(current_node)
    return list(anchors), list(deps)




def prof_stat(sess, bits, deps, feed_dict):
    def stat_func(var):
        return tf.maximum(0.0, tf.ceil(tf.log(tf.reduce_max(tf.abs(var))) / tf.log(2.0)))

    monitor_tensors = [sess.graph.get_tensor_by_name(t.name) for t in deps]
    stat_ops = [stat_func(t) for t in monitor_tensors]#
    if feed_dict != None :
        stats = sess.run(stat_ops, feed_dict) #取回运算的结果给
    else:
        stats = sess.run(stat_ops)

    print("Print Stat: ")
    for i in range(len(deps)):
        print('name:{}|{}|a:{}|L:{}|n:{}\n'.format( \
        deps[i].name, deps[i].dtype, stats[i], bits, bits-stats[i]-1)    \
        ) 
    return stats

def linear_quantize_tf_deploy(x, n, bits):
    with tf.variable_scope(x.op.name + '_quan'):
        delta = tf.pow(2.0, -n)
        bound = tf.pow(2.0, bits-1)

        min_val = - bound
        max_val = bound - 1
        x_rounded = tf.floor(x / delta + 0.5)
        return tf.clip_by_value(x_rounded, min_val, max_val) * delta
        

def add_quantize_fn(sess, bits, feed_dict, 
                    anchor_white_list, anchor_name_filter_white, 
                    anchor_name_filter_black, deps_black_list):
    graph = sess.graph

    ops, ts = find_anchors_and_deps(graph, \
                anchor_white_list=anchor_white_list,
                anchor_name_filter_white=anchor_name_filter_white, 
                anchor_name_filter_black=anchor_name_filter_black,
                deps_black_list=deps_black_list)
    for op in ops:
        print(op.name)
    '''  
    if feed_dict == None:
        stats = prof_stat(sess, bits, ts, None)
    else:
        print("++++++++",feed_dict)
        stats = prof_stat(sess, bits, ts, feed_dict)

    ts_fn = [linear_quantize_tf_deploy(ts[i], bits-stats[i]-1, bits) for i in range(len(stats))]

    ops_name = [op.name for op in ops]
    map_op_ts = {ts[i].op.name+":0": ts_fn[i] for i in range(len(ts))}
    edit_graph_fn(graph, ops_name, map_op_ts)
    return stats

    '''
    return 1
