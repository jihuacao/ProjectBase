# coding=utf-8
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--standard_component_file',
    action='store',
    dest='StandardcomponentFile',
    type=str,
    default='',
    help='the standard component config file rpath'
)
parser.add_argument(
    '--using_component_file',
    action='store',
    dest='UsingcomponentFile',
    type=str,
    default='',
    help='the using component config file rpath'
)


if __name__ == '__main__':
    args = parser.parse_args()
    using_component_file = args.UsingcomponentFile
    standard_component_file = args.StandardcomponentFile
    using_component_dict = dict()
    with open(using_component_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            split_result = line.split('=')
            if len(split_result) != 2:
                continue
                pass
            else:
                using_component, using_status = (split_result[0], split_result[1])
                pass
            if using_status not in ('ON', 'OFF'):
                continue
            using_component = using_component.replace(' ', '').replace('\n', '')
            using_status = using_status.replace(' ' , '').replace('\n', '')
            using_component_dict[using_component] = using_status
            pass
        pass
    standard_component_dict = dict()
    with open(standard_component_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            standard_component, standard_status = line.split('=')
            standard_component = standard_component.replace(' ', '').replace('\n', '')
            standard_status = standard_status.replace(' ', '').replace('\n', '')
            standard_component_dict[standard_component] = standard_status
            pass
        pass
    uis, sis = (using_component_dict.items(), standard_component_dict.items())
    for ui in uis:
        if ui[0] not in standard_component_dict.keys():
            using_component_dict.pop(ui[0])
            pass
        pass
    for si in sis:
        if si[0] not in using_component_dict.keys():
            using_component_dict[si[0]] = 'OFF'
            pass
        pass
    with open(using_component_file, 'w') as fp:
        for ui in using_component_dict.items():
            fp.write('{0}={1}\n'.format(ui[0], ui[1]))
            pass
        fp.close()
        pass
    pass