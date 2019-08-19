SHELL := /bin/bash

base:
	docker pull ros:kinetic-robot-xenial
	docker build --network=host -t aquadrone_latest -t aquadrone_base -f DockerfileBase ..

nvidia:
	docker build --network=host -t aquadrone_latest -t aquadrone_nvidia -f DockerfileNvidia --build-arg BASE_IMAGE=aquadrone_base ..
