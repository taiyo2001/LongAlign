SHELL := /bin/bash
.ONESHELL:

.PHONY: create activate remove install exec/demo

CONDA_NAME = LongAlign
PYTHON_VERSION = 3.10

create:
	conda create -y -n $(CONDA_NAME) python=$(PYTHON_VERSION) \
	&& source activate base \
	&& conda activate $(CONDA_NAME) \
	&& pip install --upgrade pip \
	&& pip install -r requirements.txt.v2

install:
	make activate && \
	pip install -r requirements.txt.v2

# eval "$(make activate)" でセッションを継承して環境を有効化可能
activate:
	source activate base && \
	conda activate $(CONDA_NAME)

remove:
	conda env remove -y -n $(CONDA_NAME)

exec/sample:
	python sample.py --prompt "A peaceful scene of a small town in winter, with snow-covered houses and trees around. The town is surrounded by mountains, and the sky is covered in clouds, creating a solemn atmosphere. In the foreground, there is a boat docked on the river, with the boat itself covered in snow. The water surface of the river is calm, reflecting the houses and trees in the distance. The roofs of the houses are covered in snow, and the windows are lit up, emitting a warm yellow light. The branches of the trees are also covered in snow, with the tips of the branches showing the blue-white color of the snow. The sky is blue, with some clouds drifting, and the sun is setting, casting a soft orange glow on the horizon. The entire scene is filled with the beauty of winter, evoking the feeling of tranquility and warmth."
