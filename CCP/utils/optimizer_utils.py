from utils.encoding import DataParallelModel


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def get_learning_rate(i_iter, warmup_steps, warmup_start_lr, learning_rate, num_steps, power):
    """Sets the learning rate throught the poly method"""
    if i_iter < warmup_steps:
        return warmup_start_lr + i_iter * (learning_rate - warmup_start_lr) / warmup_steps
    else:
        return lr_poly(learning_rate, i_iter-warmup_steps, num_steps, power)

def params_grouping(net, groups):
    if type(net) is DataParallelModel:
        net = net.module

    params_l = []
    other_params = set(net.parameters())
    for i in groups:
        param = list(eval(i).parameters())
        params_l.append({'params':param})
        other_params = other_params.difference(set(param))

    params_l.append({'params':list(other_params)})
    return params_l

def adjust_optimizer(optimizer, lr, groups):
    g = list(groups.items())
    for i in range(len(g)):
        for k,v in g[i][1].items():
            if k=='lr':
                optimizer.param_groups[i]['lr'] = lr*v
            elif k=='weight_decay':
                optimizer.param_groups[i]['weight_decay'] = v
    optimizer.param_groups[-1]['lr'] = lr
