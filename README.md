# Challenging the Performance-Interpretability Trade-off: An Evaluation of Interpretable Machine Learning Models

## Environment Setup

We provide a simple script to set up the Conda environment tailored to your operating system. To get started, ensure that Conda is installed on your system and that you have cloned this repository to your local machine.

Should you face any issues during the next setup, please ensure that you have the necessary permissions to execute the script. If needed, you can make the script executable by running:

```shell
chmod +x setup_environment.sh
```


To set up the Conda environment, execute the following command in the root directory of this project:

```shell
./setup_environment.sh
```

This command will detect your operating system and create a Conda environment with the necessary dependencies for your platform. For macOS users, this will set up an environment that is compatible with Apple Silicon. For Unix/Windows users, the script will include support for cudatoolkit if applicable.

## Datasets

Dataset names are aliased in the code as follows:

### Classification
| Dataset name | Alias  |
|:-------------|:-------|
| college      | college|
 | water       | water  |
| stroke      | stroke |
| churn       | telco  |
| recidivism  | compas |
| credit      | fico   |
| income      | adult  |
| bank        | bank   |
| airline     | airline|
| weather     | weather|

### Regression


| Dataset name | Alias |
|--------------| --- |
| car         | car   |
| student     | student|
| productivity| productivity|
| insurance   | medical|
| crimes      | crimes|
| farming     | crab|
| wine        | wine|
| bike        | bike|
| house       | housing|
| diamond     | diamond|

