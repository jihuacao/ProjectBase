# coding=utf-8
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--standard_componment_file',
    action='store',
    dest='StandardComponmentFile',
    type=str,
    default='',
    help='the standard componment config file rpath'
)
parser.add_argument(
    '--using_componment_file',
    action='store',
    dest='UsingComponmentFile',
    type=str,
    default='',
    help='the using componment config file rpath'
)


if __name__ == '__main__':
    args = parser.parse_args()
    using_componment_file = args.UsingComponmentFile
    standard_componment_file = args.StandardComponmentFile
    using_componment_dict = dict()
    with open(using_componment_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            split_result = line.split('=')
            if len(split_result) != 2:
                continue
                pass
            else:
                using_componment, using_status = (split_result[0], split_result[1])
                pass
            if using_status not in ('ON', 'OFF'):
                continue
            using_componment = using_componment.replace(' ', '').replace('\n', '')
            using_status = using_status.replace(' ' , '').replace('\n', '')
            using_componment_dict[using_componment] = using_status
            pass
        pass
    standard_componment_dict = dict()
    with open(standard_componment_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            standard_componment, standard_status = line.split('=')
            standard_componment = standard_componment.replace(' ', '').replace('\n', '')
            standard_status = standard_status.replace(' ', '').replace('\n', '')
            standard_componment_dict[standard_componment] = standard_status
            pass
        pass
    uis, sis = (using_componment_dict.items(), standard_componment_dict.items())
    for ui in uis:
        if ui[0] not in standard_componment_dict.keys():
            using_componment_dict.pop(ui[0])
            pass
        pass
    for si in sis:
        if si[0] not in using_componment_dict.keys():
            using_componment_dict[si[0]] = 'OFF'
            pass
        pass
    with open(using_componment_file, 'w') as fp:
        for ui in using_componment_dict.items():
            fp.write('{0}={1}\n'.format(ui[0], ui[1]))
            pass
        fp.close()
        pass
    pass