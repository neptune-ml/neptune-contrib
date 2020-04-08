#
# Copyright (c) 2020, Neptune Labs Sp. z o.o.
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
import os
import tempfile

import neptune
import xgboost as xgb


def neptune_callback(log_model=True,
                     log_importance=True,
                     max_num_features=None,
                     log_tree=(0,),
                     **kwargs):
    """XGBoost-monitor for Neptune experiments.

    This is XGBoost callback that automatically logs training and evaluation metrics, feature importance chart,
    visualized trees and trained Booster to Neptune.

    Make sure you created an experiment before you start XGBoost training using ``neptune.create_experiment()``
    (`check our docs <https://docs.neptune.ai/neptune-client/docs/project.html
    #neptune.projects.Project.create_experiment>`_).

    Note:
        Check Neptune documentation for the `full example <https://docs.neptune.ai/integrations/xgboost.html>`_.

    Args:
        log_model (:obj:`bool`, optional, default is ``True``):
            | Log booster to Neptune after last boosting iteration.
            | If xgb.cv, log booster for all folds
        log_importance (:obj:`bool`, optional, default is ``True``):
            | Log feature importance to Neptune as image after last boosting iteration.
            | Specify number of features using ``max_num_features`` parameter below.
        max_num_features (:obj:`int`, optional, default is ``None``):
            | Plot top ``max_num_features`` features on the importance plot.
            | If ``None``, plot all features.
        log_tree (:obj:`list` of :obj:`int`, optional, default is ``[1,]``):
            | Log specified trees to Neptune as images after last boosting iteration.
            | Default is to log first tree.
        kwargs:
            Parametrize XGBoost functions used in this callback:
            `xgboost.plot_importance <https://xgboost.readthedocs.io/en/latest/python/python_api.html
            ?highlight=plot_tree#xgboost.plot_importance>`_
            and `xgboost.to_graphviz <https://xgboost.readthedocs.io/en/latest/python/python_api.html
            ?highlight=plot_tree#xgboost.to_graphviz>`_.

    Returns:
        :obj:`callback`, function that you can pass directly to the XGBoost callbacks list, for example to the
        ``xgboost.cv()``
        (`see docs <https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=plot_tree#xgboost.cv>`_)
        or ``XGBClassifier.fit()``
        (`check docs <https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=plot_tree
        #xgboost.XGBClassifier.fit>`_).

    Note:
        If you use early stopping, make sure to log model, feature importance and trees on your own.
        Neptune logs these artifacts only after last iteration, which you may not reach because of early stop.

    Examples:
        .. code:: python3

            # Basic usage in the train function
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback()])

            # Use callback in train function, do not log model
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback(log_model=False)])

            # Use callback in train function, log five trees
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback(log_tree=[0,1,2,3,4])])

            # Use callback in train function, log top 5 features' importance chart
            xgb.train(param, dtrain, num_round, watchlist,
                      callbacks=[neptune_callback(max_num_features=5)])
    """
    try:
        neptune.get_experiment()
    except neptune.exceptions.NoExperimentContext:
        msg = 'No currently running Neptune experiment. \n'\
              'To start logging to Neptune create experiment by using: `neptune.create_experiment()`. \n'\
              'More info in the documentation: '\
              '<https://docs.neptune.ai/neptune-client/docs/project.html#neptune.projects.Project.create_experiment>.'
        raise neptune.exceptions.NeptuneException(msg)

    assert isinstance(log_model, bool),\
        'log_model must be bool, got {} instead. Check log_model parameter.'.format(type(log_model))
    assert isinstance(log_importance, bool),\
        'log_importance must be bool, got {} instead. Check log_importance parameter.'.format(type(log_importance))
    if max_num_features is not None:
        assert isinstance(max_num_features, int),\
            'max_num_features must be int, got {} instead. ' \
            'Check max_num_features parameter.'.format(type(max_num_features))
    if log_tree is not None:
        if isinstance(log_tree, tuple):
            log_tree = list(log_tree)
        assert isinstance(log_tree, list),\
            'log_tree must be list of int, got {} instead. Check log_tree parameter.'.format(type(log_tree))

    def callback(env):
        # Log metrics after iteration
        for k, v in env.evaluation_result_list:
            neptune.log_metric(k, v)

        # Log booster, end of training
        if env.iteration + 1 == env.end_iteration and log_model:
            if env.cvfolds:  # cv case
                for i, cvpack in enumerate(env.cvfolds):
                    _log_model('cv-fold-{}-bst.model'.format(i), cvpack.bst)
            else:  # train case
                _log_model('bst.model', env.model)

        # Log feature importance
        if env.iteration + 1 == env.end_iteration and log_importance:
            importance = xgb.plot_importance(env.model, max_num_features=max_num_features, **kwargs)
            neptune.log_image('feature_importance', importance.figure)

        # Log trees
        if env.iteration + 1 == env.end_iteration and log_tree:
            with tempfile.TemporaryDirectory(dir='.') as d:
                for i in log_tree:
                    file_name = 'tree_{}'.format(i)
                    tree = xgb.to_graphviz(booster=env.model, num_trees=i, **kwargs)
                    tree.render(filename=file_name, directory=d, view=False, format='png')
                    neptune.log_image('trees',
                                      os.path.join(d, '{}.png'.format(file_name)),
                                      image_name=file_name)
    return callback

# ToDo docstrings
# ToDo Check with CV
# ToDo Check with sklean API


def _log_model(name, booster):
    with tempfile.TemporaryDirectory(dir='.') as d:
        path = os.path.join(d, name)
        booster.save_model(path)
        neptune.log_artifact(path)
