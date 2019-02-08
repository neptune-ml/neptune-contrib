#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Syncs json file containg experiment data with Neptune project.

You can run your experiment in any language, create a `.json` file 
that contains your hyper parameters, metrics, tags or properties and send that to Neptune.

Attributes:
    filepath(str): filepath to the `.json` file that contains experiment data. It can have 
        ['tags', 'channels', 'properties', 'parameters', 'name'] sections.
    project_name(str): Full name of the project. E.g. "neptune-ml/neptune-examples"

Example:
    Run the experiment and create experiment json in any language.
    For example, lets say your `experiment_data.json` is
    
    >>> {"name": "baseline", 
    >>> "parameters": {"lr": 0.1, 
    >>>                "batch_size": 32
    >>>                }, 
    >>> "tags": ["base", "solution-1", "pytorch"], 
    >>> "channels": {"log_loss": {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
    >>>                           "y": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    >>>                           }
    >>>             }, 
    >>> "properties": {"data_version": "version_1"
    >>>               }
    >>> }

    Now you can sync your file with neptune. 
        $ python neptunecontrib.sync.with_json --project neptune-ml/neptune-examples
        --filepath experiment_data.json
"""

import argparse
import json
from subprocess import call


def main(args):
    with open('neptune_sync_main.py', 'w') as main:
        write_main_content(main, args.filepath)
        
    with open('neptune_sync_config.yaml', 'w') as config:
        write_config_content(config, args.filepath)
            
    call('neptune run \
        --exclude neptune_sync_main.py \
        --project {} \
        --config neptune_sync_config.yaml \
        neptune_sync_main.py'.format(args.project_name), shell=True)
    call('rm neptune_sync_config.yaml neptune_sync_main.py', shell=True)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath')
    parser.add_argument('-p', '--project_name')
    return parser.parse_args()
    
    
def write_main_content(main_file, experiment_filepath):
    main_content = """
import neptune
import json

ctx = neptune.Context()
    
with open('{}', 'r') as fp:
    data = json.load(fp)
        
for name, channel in data['channels'].items():
    for x, y in zip(channel['x'], channel['y']):
        ctx.channel_send(name, x, y)
    
for name, value in data['properties'].items():
    ctx.properties[name] = value
            
ctx.tags.extend(data['tags'])
"""
    
    main_file.write(main_content.format(experiment_filepath))
    
    
def write_config_content(config_file, experiment_filepath):
    with open(experiment_filepath, 'r') as fp:
        data = json.load(fp)
    
    config_file.write('name: {}\n\n'.format(data['name']))
    config_file.write('parameters:\n')
    for name, value in data['parameters'].items():
        config_file.write('   {}: {}\n'.format(name, value))
    
if __name__ == '__main__':
    
    args = parse_args()
    main(args)
    