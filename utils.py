import logging

import boto3
import pandas as pd
from botocore.exceptions import ClientError

import evaluator as evl
from plan import Plan

BUCKET = "balancer-results-icml"


# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        _ = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def make_plan(designs):
    plan = Plan()

    for name, dgn, estr, kw in designs:
        plan.add_design(name, dgn, estr, kw)

    plan.add_evaluator("ATEError", evl.ATEError)
    plan.add_evaluator("CovariateMSE", evl.CovariateMSE)
    plan.add_evaluator("ATECovers", evl.ATECovers)
    plan.add_evaluator("CISize", evl.CISize)
    plan.add_evaluator("AvgMarginal", evl.AvgMarginalProb)
    plan.add_evaluator("EntropyMarginal", evl.EntropyMarginalProb)
    return plan


def make_multitreat_plan(designs):
    plan = Plan()

    for name, dgn, estr, kw in designs:
        plan.add_design(name, dgn, estr, kw)

    plan.add_evaluator(
        "ATEError", evl.MultiTreatmentEvaluator, {"base_eval_class": evl.ATEError}
    )
    plan.add_evaluator(
        "CovariateMSE",
        evl.MultiTreatmentEvaluator,
        {"base_eval_class": evl.CovariateMSE},
    )
    plan.add_evaluator(
        "ATECovers", evl.MultiTreatmentEvaluator, {"base_eval_class": evl.ATECovers}
    )
    plan.add_evaluator(
        "CISize", evl.MultiTreatmentEvaluator, {"base_eval_class": evl.CISize}
    )
    plan.add_evaluator("AvgMarginal", evl.AvgMarginalProb, {})
    plan.add_evaluator("EntropyMarginal", evl.EntropyMarginalProb, {})
    return plan


def collate_and_save(dfs):
    results = pd.concat(dfs)

    filename = "results/all_results.csv.gz"

    print(f"""
**********************************************************************
***\t\tSAVING TO `{filename}`\t\t   ***
**********************************************************************
""")

    results.to_csv(filename, index=False)
    return results
