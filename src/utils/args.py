import argparse



def str2bool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise

def parse_str_to_list(string, value_type=str, sep=','):
    if string:
        outputs = string.replace(' ', '').split(sep)
    else:
        outputs = []
    
    outputs = [value_type(output) for output in outputs]

    return outputs

def parse_str_to_dict(string, value_type=str, sep=','):
    items = [s.split(':') for s in string.replace(' ', '').split(sep)]
    return {k: value_type(v) for k, v in items}

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_args_line(line):
    # Parse a value from string
    parts = line[:-1].split(': ')
    if len(parts) > 2:
        parts = [parts[0], ': '.join(parts[1:])]
    k, v = parts
    v_type = str
    if v.isdigit():
        v = int(v)
        v_type = int
    elif isfloat(v):
        v_type = float
        v = float(v)
    elif v == 'True':
        v = True
    elif v == 'False':
        v = False

    return k, v, v_type

def parse_args(args_path):
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    with open(args_path, 'rt') as args_file:
        lines = args_file.readlines()
        for line in lines:
            k, v, v_type = parse_args_line(line)
            parser.add('--%s' % k, type=v_type, default=v)

    args, _ = parser.parse_known_args()

    return args