import os
import re

def txt2list(file_src):
    """Read a text file and return its lines as a list."""
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    """Create parent directories if they don't exist."""
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    """Convert unicode to ASCII string, stripping newlines."""
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    """Check if a string contains any digit."""
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    """Remove all occurrences of specified characters from a string."""
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    """Merge two dictionaries (y overwrites x on key collision)."""
    z = x.copy()
    z.update(y)
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    """Early stopping strategy.

    Args:
        log_value: current metric value
        best_value: best metric value so far
        stopping_step: number of consecutive steps without improvement
        expected_order: 'acc' (higher is better) or 'dec' (lower is better)
        flag_step: patience threshold
    """
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
