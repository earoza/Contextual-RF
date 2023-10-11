FROM python:3.8.3

# Update the package lists and install necessary dependencies
RUN apt-get update 

################## BEGIN INSTALLATION ######################
#Install python basics
RUN apt-get -y install \
	build-essential \
	python3-dev \
	python3-setuptools \
	python3-pip

#Install scikit-learn dependancies
RUN apt-get -y install \
	python3-numpy \
	python3-scipy 

#Install scikit-learn
RUN apt-get -y install python3-sklearn

RUN pip3 install numpy==1.19.5

# Install Cython
RUN pip3 install --no-cache-dir cython>=0.28.5

################## END INSTALLATION ########################

# Copy all folders and files from the local directory to the Docker image
COPY . /app

# Set the working directory
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# Run your Python script and display the logs
CMD ["python", "main.py"]
