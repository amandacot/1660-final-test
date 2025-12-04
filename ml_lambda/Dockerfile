# Use the official AWS Lambda Python runtime base image
FROM public.ecr.aws/lambda/python:3.13

# Install system packages needed to build ML libraries
RUN dnf -y update && \
    dnf -y install gcc gcc-c++ make libgomp libffi-devel bzip2 bzip2-devel && \
    dnf clean all

# Set the working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy function code
COPY app.py .

# Copy requirements
COPY requirements.txt .

# Upgrade pip (optional, helps with latest wheel builds)
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Lambda handler
CMD ["app.lambda_handler"]
