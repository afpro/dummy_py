__all__ = [
    'file_lines_generator',
]


def file_lines_generator(path: 'str', with_eol=False):
    """
    read lines from file
    :param path: path
    :param with_eol: where line with '\n' suffix
    :return: file line generator
    """
    with open(path) as fp:
        while True:
            line = fp.readline()
            if len(line) == 0:
                break
            if with_eol:
                yield line
            else:
                yield line[:-1]
