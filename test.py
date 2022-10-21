import random
import numpy as np 
import itertools

config = {'max_num_iteration': 3000,\
                   'ds':10,\
                   'mutation_probability':0.1,\
                   'elite_ratio': 0.5,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None,
                   "attrs": 6,
                   "goals": [1,1]
                   }


change_list = np.zeros(config["attrs"])
change_list[0] = 1
change_list[2] = 1
change_list[3] = 1
change_list[4] = 1
print(change_list)

change_list_inverse = np.ones_like(change_list) - change_list

column_config = np.zeros((config["attrs"], 2))
for i in range(len(column_config)):
    column_config[i][1] = 1

def make_crossover_config():
    length = int(np.sum(change_list))
    tmp = np.zeros(length)
    tmp2 = list(np.arange(length))
    np.random.permutation(tmp2)
    ind1 = tmp2.pop()
    ind2 = tmp2.pop()

    config1 = np.zeros(length)
    #config2 = np.zeros(np.sum(change_list))

    for i in range(0, min(ind1, ind2)):
        config1[i] = 1

    #for i in range(min(ind1, ind2), max(ind1, ind2)):
    #    config2[i] = 1
    
    for i in range(max(ind1, ind2), length):
        config1[i] = 1
    
    return config1, np.ones_like(config1) - config1


co_config, co_config_inv = make_crossover_config()




    

def make_data(xs, attrs, ds):
    w = np.random.rand(attrs)
    b = np.random.rand(1)
    w = w.reshape(-1, 1)
    y_hat_tmp = np.dot(xs, w)
    y_hat = y_hat_tmp + b[0]
    assert y_hat.shape[0] == ds
    return w, b, y_hat

def get_initial_random_data():
    return np.random.rand(config["ds"], config["attrs"])

def predict_y_hat(xs, w, b):
    return np.dot(xs, w) + b

def extract_elites(xs, y_hat, goals):
    assert y_hat.shape[1] == len(goals)
    loss_tmp = y_hat - goals
    loss = loss_tmp * loss_tmp
    loss_sum = np.sum(loss, axis=-1)
    elite_indices = np.argsort(loss_sum)

    elite_num = int(config["ds"] * config["elite_ratio"])
    print(elite_num)

    return xs[elite_indices[:elite_num]]


def make_next_generation(elites):
    res = []
    num = len(elites)
    pairs = list(itertools.combinations(np.arange(num), 2))
    for i, pair in enumerate(pairs):
        print(f"{i} / {len(pairs)} ======================")
        mutated_child1, mutated_child2 = make_child_main(elites[pair[0]], elites[pair[1]])
        res.append(mutated_child1)
        res.append(mutated_child2)
    return res


def extract_threshold_target_from_data(xs):
    #tmp = xs[:, np.where(co_config==1)[0]]
    tmp = xs[np.where(change_list==1)[0]]
    return tmp

def make_masked_data_for_fixing(xs):
    return xs * change_list_inverse

def put_back_child_to_masked_data_for_fixing(child, masked_data):
    tmp = np.where(change_list==1)[0]
    for i in range(len(tmp)):
        masked_data[tmp[i]] = child[i]
    
    return masked_data




def apply_cross_over(parent1, parent2):

    if random.random() > config["crossover_probability"]:
        res1 = parent1 * co_config + parent2 * co_config_inv
        res2 = parent2 * co_config + parent1 * co_config_inv
        return res1, res2
    else:
        return parent1, parent2


def apply_mutation(child):
    res = np.zeros_like(child)
    for i in range(len(child)):
        if random.random() > config["mutation_probability"]:
            res[i] = child[i]
        else:
            res[i] = random.random()
    
    return res


def make_child_main(parent1, parent2):
    # parent in with full length
    parent1_extracted = extract_threshold_target_from_data(parent1)
    parent2_extracted = extract_threshold_target_from_data(parent2)

    parent1_masked_for_fix = make_masked_data_for_fixing(parent1)
    parent2_masked_for_fix = make_masked_data_for_fixing(parent2)

    child1, child2 = apply_cross_over(parent1_extracted, parent2_extracted)
    mutated_child1 = apply_mutation(child1)
    mutated_child2 = apply_mutation(child2)
    next_gen1 = put_back_child_to_masked_data_for_fixing(mutated_child1, parent1_masked_for_fix)
    next_gen2 = put_back_child_to_masked_data_for_fixing(mutated_child2, parent2_masked_for_fix)

    return next_gen1, next_gen2



if __name__ == "__main__":

    print("hello, world")

    xs = np.random.rand(config["ds"], config["attrs"])
    w1_true, b1_true, y1 = make_data(xs, config["attrs"], config["ds"])
    w2_true, b2_true, y2 = make_data(xs, config["attrs"], config["ds"])

    y_hat1 = predict_y_hat(xs, w1_true, b1_true)
    y_hat2 = predict_y_hat(xs, w2_true, b2_true)
    y_hat_total = np.array([y_hat1, y_hat2])

    a = np.random.rand(config["ds"], config["attrs"])
    b = a[:, np.where(co_config==1)[0]]
    print("here")
    print(b.shape)
    print(co_config)

    




    y_hat_np = np.zeros((xs.shape[0], len(config["goals"])))
    for i in range(xs.shape[0]):
        for j in range(len(config["goals"])):
            y_hat_np[i][j] = y_hat_total[j][i]
    print(y_hat_np)

    elites = extract_elites(xs, y_hat_np, config["goals"])
    next_gens = make_next_generation(elites)
    print("=====================")
    print(next_gens)


    
