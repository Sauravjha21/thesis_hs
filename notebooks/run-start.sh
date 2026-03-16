#!/bin/bash

set -e

sudo -u ec2-user -i <<'EOF'

ENVIRONMENT=python3

conda activate "$ENVIRONMENT"

pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade sklearn
pip install --upgrade matplotlib
pip install --upgrade scipy
pip install --upgrade statsmodels
pip install --upgrade dataclasses
pip install --upgrade boto3

conda deactivate

IDLE_TIME=3600

echo "Fetching the autostop script"
wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py

echo "Starting the SageMaker autostop script in cron"

(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab -

EOF
