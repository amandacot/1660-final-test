# Use AWS Lambda Python 3.13 base image
FROM public.ecr.aws/lambda/python:3.13

# Install system dependencies for cffi and compiling packages
RUN dnf -y update && \
    dnf -y install gcc gcc-c++ make libffi-devel bzip2 bzip2-devel && \
    dnf clean all

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy your Lambda function code
COPY app.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler (data.lambda_handler)
CMD ["app.lambda_handler"]
