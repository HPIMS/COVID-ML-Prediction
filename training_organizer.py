#!/bin/python

import os

import pandas as pd
import numpy as np

from config import Config
from train_test_model import HammerTime


class Organizer:
    def __init__(self, filename):
        os.makedirs(Config.results_dir, exist_ok=True)
        self.df = pd.read_pickle(filename)

        # Rename features - Hyphens cause issues
        columns = [
            i.replace(' ', '').replace('-', '_')
            for i in self.df.columns]
        self.df.columns = columns

        # This is finicky
        self.outcome = filename.split('/')[1].split('_')[0]
        self.day = filename.split('/')[1].split('_')[1].replace('.pickle', '')

        # Metabolite dataframes
        self.df_train_test = None
        self.df_prospective = None

        self.df_msh = None
        self.df_oh = None
        self.df_prospective_msh = None
        self.df_prospective_oh = None

    def organize_data(self):
        # Separate into training / testing / validation cohorts
        # on the basis of Admission date
        separation_date = pd.to_datetime(Config.cutoff_date)
        self.df_train_test = self.df.query('Admit_Date < @separation_date')
        self.df_prospective = self.df.query('Admit_Date >= @separation_date')

    def train_test_validate_model(self):
        # Subdivide df_train_test on the basis of Last_Facility
        # Cross validate on df_msh / Test on df_oh
        self.df_msh = self.df_train_test.query('Last_Facility == "MSH"')
        self.df_oh = self.df_train_test.query('Last_Facility != "MSH"')

        self.df_prospective_msh = self.df_prospective.query('Last_Facility == "MSH"')
        self.df_prospective_oh = self.df_prospective.query('Last_Facility != "MSH"')

        # Doesn't want to work inside a loop
        self.df_msh = self.df_msh.drop(['Admit_Date', 'Last_Facility'], axis=1)
        self.df_oh = self.df_oh.drop(['Admit_Date', 'Last_Facility'], axis=1)
        self.df_prospective_msh = self.df_prospective_msh.drop(['Admit_Date', 'Last_Facility'], axis=1)
        self.df_prospective_oh = self.df_prospective_oh.drop(['Admit_Date', 'Last_Facility'], axis=1)

        # Show sizes of each cohort
        print(
            'MSH:', self.df_msh.shape[0],
            'OH:', self.df_oh.shape[0],
            'PT_MSH:', self.df_prospective_msh.shape[0],
            'PT_OH:', self.df_prospective_oh.shape[0])

        # Cross validation is automatic
        print('Cross validating...')
        hammer_time = HammerTime(self.df_msh, self.outcome, self.day)

        # Save thresholds - have to be rounded up
        pd.to_pickle(
            hammer_time.dict_thresholds,
            f'Results/thresh_{self.outcome}_{self.day}',
            protocol=4)

        # Additional testing and validation have to be called
        print('Testing and validating...')
        hammer_time.test_validate(self.df_oh, 'test')
        hammer_time.test_validate(self.df_prospective_msh, 'pt_msh')
        hammer_time.test_validate(self.df_prospective_oh, 'pt_oh')

        self.aggregate_metrics(hammer_time.dict_metrics)

    def calc_sens_spec(self, cm):
        sens = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        spec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        return sens, spec

    def aggregate_metrics(self, dict_metrics):

        # Save raw metrics
        outcome = f'{self.outcome}_{self.day}'
        outfile = f'raw_{outcome}.pickle'
        pd.to_pickle(dict_metrics, os.path.join(Config.results_dir, outfile))

        # df from here
        all_metrics = []

        # This is left redundant on purpose
        # Cross val
        for clf_desc in Config.clf_descriptions:

            desc = 'MSH > MSH'
            patients = self.df_msh.shape[0]
            outcome_perc = self.df_msh[self.outcome].sum() / patients
            acc = np.mean(dict_metrics['cross_val'][clf_desc]['acc'])
            auroc = np.mean(dict_metrics['cross_val'][clf_desc]['auroc'])
            auprc = np.mean(dict_metrics['cross_val'][clf_desc]['auprc'])
            f1s = np.mean(dict_metrics['cross_val'][clf_desc]['f1s'])

            cm = dict_metrics['cross_val'][clf_desc]['c_matrix'][0]
            for arr in dict_metrics['cross_val'][clf_desc]['c_matrix'][1:]:
                cm += arr
            sens, spec = self.calc_sens_spec(cm)

            metrics = [
                outcome, desc, clf_desc, patients, outcome_perc,
                acc, auroc, auprc, f1s, sens, spec]
            all_metrics.append(metrics)

            # Better descriptions
            dict_desc = {
                'test': ['MSH > OH', self.df_oh],
                'pt_msh': ['MSH > PROSPECTIVE MSH', self.df_prospective_msh],
                'pt_oh': ['MSH > PROSPECTIVE OH', self.df_prospective_oh]
            }

        # Test / validate
        clf_desc = 'XGB_NotImputed'
        for key in dict_desc:
            desc = dict_desc[key][0]
            patients = dict_desc[key][1].shape[0]
            outcome_perc = dict_desc[key][1][self.outcome].sum() / patients
            acc = dict_metrics[key][clf_desc]['acc'][0]
            auroc = dict_metrics[key][clf_desc]['auroc'][0]
            auprc = dict_metrics[key][clf_desc]['auprc'][0]
            f1s = dict_metrics[key][clf_desc]['f1s'][0]

            cm = dict_metrics[key][clf_desc]['c_matrix'][0]
            sens, spec = self.calc_sens_spec(cm)

            metrics = [
                outcome, desc, clf_desc, patients, outcome_perc, acc,
                auroc, auprc, f1s, sens, spec]
            all_metrics.append(metrics)

        df_metrics = pd.DataFrame(all_metrics)
        df_metrics = df_metrics.round(3)
        df_metrics.columns = [
            'OUTCOME', 'OPERATION', 'CLASSIFIER', 'PATIENTS', 'OUTCOME PERC',
            'ACCURACY', 'AUROC', 'AUPRC', 'F1S', 'SENS', 'SPEC']

        df_metrics.to_pickle(
            f'{Config.results_dir}/agg_{outcome}.pickle', protocol=4)

    def start(self):
        self.organize_data()
        self.train_test_validate_model()
