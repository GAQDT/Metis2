import re
import ast
import sys

def split_row(line):
    fields = []
    current = []
    bracket_level = 0
    for char in line.strip():
        if char == ',' and bracket_level == 0:
            fields.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
            if char in '([{':
                bracket_level += 1
            elif char in ')]}':
                bracket_level -= 1
    fields.append(''.join(current).strip())
    return fields

use_full = len(sys.argv) > 1 and sys.argv[1] == '1'
input_file = '/root/codes/MetisOnAlpa/full_estimate_costs.txt' if use_full else '/root/codes/MetisOnAlpa/estimate_costs.txt'
with open(input_file , 'r') as f:
    lines = f.readlines()

gbs = int(lines[0].strip())
model_name = lines[1].strip()

nodes = {}
if lines and lines[2].startswith('['):
    try:
        node_list = ast.literal_eval(lines[2].strip())
        for idx, node_info in enumerate(node_list):
            if len(node_info) == 3 and node_info[0] == 'node':
                nodes[idx] = {
                    'device_type': node_info[1].lower(),
                    'num_devices': int(node_info[2])
                }
    except Exception as e:
        print(f"Error parsing node info: {e}")

header_line = lines[3].strip()
headers = [h.split('(')[0].strip() for h in split_row(header_line)]

data = []
for line in lines[4:]:
    line = line.strip()
    if not line:
        continue
    fields = split_row(line)
    if len(fields) != len(headers):
        print(f"Skipping invalid line: {line}")
        continue
    try:
        record = {
            'rank': int(fields[0]),
            'cost': float(fields[1]),
            'node_sequence': tuple(re.findall(r"'([^']*)'", fields[2])),
            'device_groups': ast.literal_eval(fields[3]),
            'strategies': ast.literal_eval(fields[4]),
            'batches': int(fields[5]),
            'layer_partition': ast.literal_eval(fields[6])
        }
        data.append(record)
    except Exception as e:
        print(f"Error processing line: {line}\nError: {e}")
        continue

print("\nParsed GPU nodes:")
for node_id, info in nodes.items():
    print(f"Node {node_id}: {info['device_type']} x{info['num_devices']}")
print(f"Parsed {len(data)} records")

######################################################

plans = []

for i in range(len(data)):
    forward_stage_layer_ids = []
    submesh_physical_shapes = []
    submesh_logical_shapes = []
    submesh_to_hosts = []

    for j in range(len(data[i]['layer_partition']) - 1):
        forward_stage_layer_ids.append(list(range(
            data[i]['layer_partition'][j],
            data[i]['layer_partition'][j + 1]
        )))

    submesh_logical_shapes = data[i]['strategies']
    node_idx = 0
    node_used = 0
    for j in range(len(data[i]['device_groups'])):
        group_size = data[i]['device_groups'][j]
        if group_size < nodes[node_idx]['num_devices'] - node_used:
            submesh_physical_shapes.append((1, group_size))
            node_used += group_size
        elif group_size == nodes[node_idx]['num_devices'] - node_used:
            submesh_physical_shapes.append((1, group_size))
            node_used = 0
            node_idx += 1
        else:
            k = group_size / nodes[node_idx]['num_devices']
            submesh_physical_shapes.append((int(k), nodes[node_idx]['num_devices']))
            node_idx += int(k)

    plan = {
        'forward_stage_layer_ids': forward_stage_layer_ids,
        'submesh_physical_shapes': submesh_physical_shapes,
        'submesh_logical_shapes': submesh_logical_shapes,
        'submesh_to_hosts': submesh_to_hosts,
        'batches': data[i]['batches'],
        'layers': data[i]['layer_partition'][-1],
        'node_sequence': data[i]['node_sequence']
    }
    plans.append(plan)

output_file = '/root/codes/MetisOnAlpa/full_plans.txt' if use_full else '/root/codes/MetisOnAlpa/plans.txt'

with open(output_file, 'w') as f:
    f.write(f'{gbs}\n')
    f.write(f'{model_name}\n') 
    for plan in plans:
        f.write(f'{plan}\n')
