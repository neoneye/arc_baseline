# Docker baseline

This repo is an example docker solution for the challenge https://lab42.global/arcathon/.
The main python code is a copied version from https://www.kaggle.com/code/user189546/5-crop-tasks-by-brute-force.

## I. Solution structure
```
.
├── code
│             ├── abstraction-and-reasoning-challenge.zip
│             ├── arc_crop.py
│             ├── arc_post.py
│             ├── arc_pre.py
│             └── run.sh
├── data
│             ├── evaluation
│             │             ├── 0bb8deee.json
│             │             ├── 0becf7df.json
│             │             └── 1a6449f1.json
│             └── solution
│                 ├── PLACE_HOLDER.txt
│                 ├── solution_armo.json
├── Dockerfile
└── README.md
```

- The code folder contains your scripts, python codes ...
- The data folder contains public tasks to develope and test your solution.
Note that you should adapt the json filename to include your team name. My team name is `armo`.

## II. Docker image creation

- Build docker image: `docker build -t public_docker .`
- Verify the code: `docker run --mount type=bind,source="$(pwd)"/secret_data,target=/data -e token=public_token public_docker`

You should be able to see the solution json file in the data/solution directory

## III. Docker publication
Note: `saimo2020` is my docker hub account, you should replace with your own account.

- Tag your docker image: `docker tag public_docker:latest saimo2020/public_docker:v11`
- Push your docker image: `docker image push saimo2020/public_docker:v11`

For the first time, it would take a while to push different layers in the docker hub. Once the base layers are published, the next push should take only few seconds.

You should be able see your docker image is published in the docker hub
https://hub.docker.com/repository/docker/saimo2020/public_docker/general

## IV. Submission

- Locate to the website: https://lab42.global/arcathon/submission/
- Docker-related fields
  - `docker pull saimo2020/public_docker:v11`
  - `docker run --mount type=bind,source="$(pwd)"/secret_data,target=/data saimo2020/public_docker:v11`
