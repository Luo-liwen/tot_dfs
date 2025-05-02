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

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

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

# def check_steps(new_ys):
#     cleaned_ys = []
#     for idx, item in enumerate(new_ys):
#         lines = item.strip().split('\n')
#         if len(lines) > 1:
#             # 获取倒数第二行的left列表
#             last_step_left = re.search(r'left:\s*(.*)\)', lines[-2])
#             if last_step_left:
#                 last_step_left = [float(x) for x in last_step_left.group(1).split() if x]
#             else:
#                 print(f'第{idx}个 item 找不到上一行 left')
#                 continue

#             # 获取最后一行式子和结果
#             this_step = lines[-1]

#             # 匹配表达式和结果
#             expr_match = re.match(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)', this_step)

#             if not expr_match:
#                 print(f'第{idx}个 item 最后一行格式不对: {this_step}')
#                 continue

#             num1_str, op, num2_str, result_str = expr_match.groups()
#             num1, num2, result_part = float(num1_str), float(num2_str), float(result_str)

#             # 检查操作数是否都在last_step_left中
#             if num1 not in last_step_left or num2 not in last_step_left:
#                 print(f'第{idx}个 item 错误：{num1}, {num2} 不在上一步 {last_step_left} 中')
#                 continue


#             # 获取最后一行的left
#             new_left_match = re.search(r'left:\s*(.*)\)', this_step)
#             if new_left_match:
#                 new_left = [float(x) for x in new_left_match.group(1).split() if x]
#             else:
#                 print(f'第{idx}个 item 找不到当前行 left')
#                 continue

#             # 构造预期left
#             expected_left = last_step_left.copy()
#             try:
#                 expected_left.remove(num1)
#                 expected_left.remove(num2)
#             except ValueError:
#                 print(f'第{idx}个 item 删除元素时出错：{num1}, {num2}, 当前 {expected_left}')
#                 continue
#             expected_left.append(result_part)

#             # 排序后对比
#             if sorted(expected_left) != sorted(new_left):
#                 print(f'第{idx}个 item left错误：当前 {new_left} ≠ 预期 {expected_left}')
#                 continue
#         cleaned_ys.append(item)
#     return cleaned_ys


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



def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        if step>0 and step<task.steps-1:
            new_ys=check_steps(new_ys)
        
        
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
            
            
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        if select_new_ys == []:
            return [ys[0]], {'steps': infos} #只返回一个错的结果
        
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}