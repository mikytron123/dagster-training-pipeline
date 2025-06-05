<div align="center">

  <h3 align="center">Dagster-training-pipeline</h3>

  <p align="center">
ML model training pipeline using dagster and mlflow
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project is a machine learning training pipeline that is orchestrated using Dagster. A postgres database and minio instance is used for storage. MLFlow is used for experiment tracking of different machine learning models.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

![Diagram](images/Job_train_model_pipeline_final.svg)

### Built With


* [![postgresql][postgresql-logo]][postgresql-url]
* [![rabbitmq][rabbitmq-logo]][rabbitmq-url]
* [![dagster][dagster-logo]][dagster-url]
* [![minio][minio-logo]][minio-url]
* [![mlflow][mlflow-logo]][mlflow-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these steps.

### Prerequisites

Install Docker. Create a .env file using env.example as an example.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/mikytron123/dagster-training-pipeline.git
   ```
2. Launch docker containers
   ```sh
   docker compose up
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Navigate to `localhost:3000` to view the dagster ui and run the pipeline.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[postgresql-logo]: https://img.shields.io/badge/postgresql-4169E1?style=for-the-badge&logo=postgresql&logoColor=white
[postgresql-url]: https://wiki.postgresql.org/wiki/Logo
[rabbitmq-logo]: https://img.shields.io/badge/rabbitmq-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white
[rabbitmq-url]: https://www.rabbitmq.com
[dagster-logo]: https://img.shields.io/badge/dagster-black?style=for-the-badge
[dagster-url]: https://dagster.io/
[minio-logo]: https://img.shields.io/badge/minio-C72E49?style=for-the-badge&logo=minio&logoColor=white
[minio-url]: https://min.io
[mlflow-logo]: https://img.shields.io/badge/mlflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white
[mlflow-url]: https://github.com/mlflow/mlflow/blob/855881f93703b15ffe643003fb4d7c84f0ec2502/assets/icon.svg


