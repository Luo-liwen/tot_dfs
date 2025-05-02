import itertools
import numpy as np
from functools import partial
from src.tot.models import gpt
import re


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def should_prune(value, current_step, best_value, pruning_threshold):
    # 如果当前值比最好值差太多，就剪枝
    if best_value['value'] != float('-inf'):
        if value < best_value['value'] * pruning_threshold:
            return True
    # 也可以根据当前步骤设置不同的阈值
    # step_thresholds = {
    #     0: 1,  # 第一步要求不那么严格
    #     1: 3,  # 第二步更严格
    #     2: 3,   # 第三步最严格
    #     3:3
    # }
    step_thresholds = {
        0: 3,  # 第一步要求不那么严格
        1: 3,  # 第二步更严格
        2: 3,   # 第三步最严格
        3:3
    }
    if current_step in step_thresholds:
        if value < step_thresholds[current_step]:
            return True
    return False

def safe_float(s):
    s = s.replace('...', '')  # 去掉省略号
    try:
        return float(s)
    except ValueError:
        print(f'无法转换的值: {s}')
        return None  # 或 raise，看你要不要跳过还是报错



def check_steps(new_ys):
    cleaned_ys = []
    for idx, item in enumerate(new_ys):
        lines = item.strip().split('\n')
        if len(lines) > 1:
            # 获取倒数第二行的left列表
            last_step_left = re.search(r'left:\s*(.*)\)', lines[-2])
            if last_step_left:
                last_step_left = [float(x) for x in last_step_left.group(1).split() if x]
            else:
                print(f'第{idx}个 item 找不到上一行 left')
                continue

            # 获取最后一行式子和结果
            this_step = lines[-1]

            # 匹配表达式和结果
            expr_match = re.match(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)', this_step)

            if not expr_match:
                print(f'第{idx}个 item 最后一行格式不对: {this_step}')
                continue

            num1_str, op, num2_str, result_str = expr_match.groups()
            num1, num2, result_part = float(num1_str), float(num2_str), float(result_str)

            # 检查操作数是否都在last_step_left中
            if num1 not in last_step_left or num2 not in last_step_left:
                print(f'第{idx}个 item 错误：{num1}, {num2} 不在上一步 {last_step_left} 中')
                continue

            # 计算表达式结果，检查是否正确
            if op == '+':
                calc_result = num1 + num2
            elif op == '-':
                calc_result = num1 - num2
            elif op == '*':
                calc_result = num1 * num2
            elif op == '/':
                calc_result = num1 / num2
            else:
                print(f'第{idx}个 item 未知操作符: {op}')
                continue

            if calc_result != result_part:
                print(f'第{idx}个 item 结果错误：{num1}{op}{num2}={calc_result} ≠ {result_part}')
                continue

            # 获取最后一行的left
            new_left_match = re.search(r'left:\s*(.*)\)', this_step)
            if new_left_match:
                # new_left = [float(x) for x in new_left_match.group(1).split() if x]
                new_left_raw = new_left_match.group(1).split()
                new_left = []
                for x in new_left_raw:
                    x_clean = x.strip().rstrip(',')  # 去掉空格和末尾逗号
                    if x_clean:
                        try:
                            new_left.append(float(x_clean))
                        except ValueError:
                            print(f"warning: could not convert '{x_clean}' to float")
                            continue
            else:
                print(f'第{idx}个 item 找不到当前行 left')
                continue

            # 构造预期left
            expected_left = last_step_left.copy()
            try:
                expected_left.remove(num1)
                expected_left.remove(num2)
            except ValueError:
                print(f'第{idx}个 item 删除元素时出错：{num1}, {num2}, 当前 {expected_left}')
                continue
            expected_left.append(result_part)

            # 排序后对比
            if sorted(expected_left) != sorted(new_left):
                print(f'第{idx}个 item left错误：当前 {new_left} ≠ 预期 {expected_left}')
                continue
        cleaned_ys.append(item)
    return cleaned_ys


# def dfs_solve_recursive(task, x, y, current_step, args, infos, best_value, to_print=True):
#     if current_step >= task.steps:
#         value = get_value(task, x, y, args.n_evaluate_sample)
#         if value > best_value['value']:
#             best_value['value'] = value
#             best_value['solution'] = y
#         return
    
#     # generation
#     if args.method_generate == 'sample':
#         new_ys = get_samples(task, x, y, args.n_generate_sample, 
#                            prompt_sample=args.prompt_sample, 
#                            stop=task.stops[current_step])
#     elif args.method_generate == 'propose':
#         new_ys = get_proposals(task, x, y)
    
#     if current_step>0 and current_step<task.steps-1:
#         new_ys = check_steps(new_ys)
        
#     # evaluation
#     values = []
#     valid_candidates = []
#     for new_y in new_ys:
#         value = get_value(task, x, new_y, args.n_evaluate_sample)
#         # 剪枝判断
#         if not should_prune(value, current_step, best_value, pruning_threshold=0.8):
#             values.append(value)
#             valid_candidates.append(new_y)
    
#     if not valid_candidates:  # 如果所有候选都被剪掉了
#         return
    
#     # selection from valid candidates
#     ids = list(range(len(valid_candidates)))
#     if args.method_select == 'sample':
#         ps = np.array(values) / sum(values)
#         select_ids = np.random.choice(ids, size=min(args.n_select_sample, len(ids)), p=ps).tolist()
#     elif args.method_select == 'greedy':
#         select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:min(args.n_select_sample, len(ids))]
    
#     select_new_ys = [valid_candidates[select_id] for select_id in select_ids]
    
#     # log
#     if to_print:
#         sorted_pairs = sorted(zip(valid_candidates, values), key=lambda x: x[1], reverse=True)
#         if sorted_pairs:
#             sorted_new_ys, sorted_values = zip(*sorted_pairs)
#             print(f'Step {current_step}:')
#             print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
    
#     # record step info
#     infos.append({
#         'step': current_step,
#         'x': x,
#         'ys': [y],
#         'new_ys': valid_candidates,
#         'values': values,
#         'select_new_ys': select_new_ys
#     })
    
#     # recursively explore the best candidates
#     for new_y in select_new_ys:
#         dfs_solve_recursive(task, x, new_y, current_step + 1, args, infos, best_value, to_print)

# def solve(args, task, idx, to_print=True):
#     global gpt
#     gpt = partial(gpt, model=args.backend, temperature=args.temperature)
#     print(gpt)
#     x = task.get_input(idx)
    
#     # Initialize info collector and best value tracker
#     infos = []
#     best_value = {
#         'value': float('-inf'),
#         'solution': None
#     }
    
#     # Start DFS from empty solution
#     dfs_solve_recursive(task, x, '', 0, args, infos, best_value, to_print)
    
#     # Return the final candidates from the last step
#     final_ys = infos[-1]['select_new_ys'] if infos else ['']
    
#     if to_print:
#         print(f"Best solution found: {best_value['solution']}")
#         print(f"Best value: {best_value['value']}")
#         print(f"Final candidates: {final_ys}")
    
#     return final_ys, {'steps': infos}

def dfs_solve_recursive(task, x, y, current_step, args, infos, best_value, final_solutions, to_print=True):
    if current_step >= task.steps:
        value = get_value(task, x, y, args.n_evaluate_sample)
        final_solutions.append((y, value))  # 把路径和对应value存进去
        if value > best_value['value']:
            best_value['value'] = value
            best_value['solution'] = y
        return
    
    if args.method_generate == 'sample':
        new_ys = get_samples(task, x, y, args.n_generate_sample, 
                             prompt_sample=args.prompt_sample, 
                             stop=task.stops[current_step])
    elif args.method_generate == 'propose':
        new_ys = get_proposals(task, x, y)
    
    # if current_step > 0 and current_step < task.steps - 1:
    #     new_ys = check_steps(new_ys)
    
    values = []
    valid_candidates = []
    for new_y in new_ys:
        value = get_value(task, x, new_y, args.n_evaluate_sample)
        if not should_prune(value, current_step, best_value, pruning_threshold=0.8):
            values.append(value)
            valid_candidates.append(new_y)
    
    if not valid_candidates:
        return
    
    # ids = list(range(len(valid_candidates)))
    # if args.method_select == 'sample':
    #     ps = np.array(values) / sum(values)
    #     select_ids = np.random.choice(ids, size=min(args.n_select_sample, len(ids)), p=ps).tolist()
    # elif args.method_select == 'greedy':
    #     select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:min(args.n_select_sample, len(ids))]
    
    select_new_ys =valid_candidates
    
    # [valid_candidates[select_id] for select_id in select_ids]
    
    if to_print:
        sorted_pairs = sorted(zip(valid_candidates, values), key=lambda x: x[1], reverse=True)
        if sorted_pairs:
            sorted_new_ys, sorted_values = zip(*sorted_pairs)
            print(f'Step {current_step}:')
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
    
    infos.append({
        'step': current_step,
        'x': x,
        'ys': [y],
        'new_ys': valid_candidates,
        'values': values,
        'select_new_ys': select_new_ys
    })
    
    for new_y in select_new_ys:
        dfs_solve_recursive(task, x, new_y, current_step + 1, args, infos, best_value, final_solutions, to_print)


def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)
    
    infos = []
    best_value = {'value': float('-inf'), 'solution': None}
    final_solutions = []  # 新增：所有最终候选
    
    dfs_solve_recursive(task, x, '', 0, args, infos, best_value, final_solutions, to_print)
    
    if final_solutions:
        # 排序，取前5个
        final_solutions.sort(key=lambda x: x[1], reverse=True)
        top_k_solutions = final_solutions[:5]
        final_ys = [item[0] for item in top_k_solutions]
    else:
        final_ys = ['']
    
    if to_print:
        print(f"Best solution found: {best_value['solution']}")
        print(f"Best value: {best_value['value']}")
        print(f"Top-{len(final_ys)} candidates: {final_ys}")
    
    return final_ys, {'steps': infos, 'all_final_solutions': final_solutions}



def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}