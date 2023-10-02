### AWS Lambda
# Use public ECR provided Python Runtime for AWS Lambda
FROM public.ecr.aws/lambda/python:3.9

# Set the working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements.txt
COPY requirements.txt .

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY app.py .
COPY config.py .

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]