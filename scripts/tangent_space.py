"""
This file classify SSVEP through MOABB library 
Compares CCA method with Riemann Geometry + Logistic Regression

This file do:
- Uses more than one subject
- Uses SciKit pipeline, passing 'RG + LogReg' param (Riemann Geometry + Logistic Regression)
--- Pipeline of (MOABB) ExtendedSSVEPSignal -> (pyRi) Covariances -> pyRi TangentSpace -> (sklearn) LogisticRegression
- Uses SciKit pipeline, passing 'CCA' param 
--- pass only (MOABB) SSVEP_CCA
- Uses (MOABB) CrossSubjectEvaluation to evaluate results
"""

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances

from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import SSVEP, FilterBankSSVEP
from moabb.datasets import SSVEPExo
from moabb.pipelines import SSVEP_CCA, ExtendedSSVEPSignal
import moabb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
moabb.set_log_level('info')

###############################################################################
# Loading dataset
# ---------------
#


# for i in range(n_subject):
#     SSVEPExo()._get_single_subject_data(i + 1)
dataset = SSVEPExo(sessions_per_subject=5)
dataset.data_path(1, path="/home/lucas-c/workspace/databases/ssvep_exo")
interval = dataset.interval
paradigm = SSVEP(fmin=10, fmax=25, n_classes=3)
paradigm_fb = FilterBankSSVEP(filters=None, n_classes=3)

# Classes are defined by the frequency of the stimulation, here we use
# the first two frequencies of the dataset, 13 and 17 Hz.
# The evaluation function uses a LabelEncoder, transforming them
# to 0 and 1

freqs = paradigm.used_events(dataset)

##############################################################################
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
# The first pipeline uses Riemannian geometry, by building an extended
# covariance matrices from the signal filtered around the considered
# frequency and applying a logistic regression in the tangent plane.
# The second pipeline relies on the above defined CCA classifier.

pipelines_fb = {}
pipelines_fb['RG + LogReg'] = make_pipeline(
    ExtendedSSVEPSignal(),
    Covariances(estimator='lwf'),
    TangentSpace(),
    LogisticRegression(solver='lbfgs', multi_class='auto'))

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (SSVEP) and use the dataset available for it.
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

overwrite = False  # set to True if we want to overwrite cached results

# Filter bank processing, determine automatically the filter from the
# stimulation frequency values of events.
evaluation_fb = CrossSubjectEvaluation(paradigm=paradigm_fb,
                                       datasets=dataset, overwrite=overwrite)
results_fb = evaluation_fb.process(pipelines_fb)

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results.

fig, ax = plt.subplots(facecolor='white', figsize=[8, 4])
sns.stripplot(data=results_fb, y='score', x='pipeline', ax=ax, jitter=True,
              alpha=.5, zorder=1, palette="Set1")
sns.pointplot(data=results_fb, y='score', x='pipeline', ax=ax,
              zorder=1, palette="Set1")
ax.set_ylabel('Accuracy')
ax.set_ylim(0.1, 0.6)
plt.savefig('ssvep.png')
fig.show()

print(results_fb)