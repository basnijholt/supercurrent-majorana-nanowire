BASE_DIR = tmp_make
CODE_DIR = $(BASE_DIR)/code

clean:
	-rm -rf $(BASE_DIR)

code:
	-mkdir -p $(BASE_DIR)
	-mkdir -p $(CODE_DIR)
	cp README.md $(CODE_DIR)/README.md
	cp LICENSE $(CODE_DIR)/
	cp funcs.py $(CODE_DIR)/
	cp common.py $(CODE_DIR)/
	cp combine.py $(CODE_DIR)/
	cp generate-data.ipynb $(CODE_DIR)/
	cp explore-data.ipynb $(CODE_DIR)/
	cp paper-figures.ipynb $(CODE_DIR)/
	cp mean-free-path.ipynb $(CODE_DIR)/
	cp example-toy-models.ipynb $(CODE_DIR)/
	cp environment.yml $(CODE_DIR)/

data: code
	$(eval TARGET_DIR := $(BASE_DIR)/data_sub)
	-mkdir -p $(TARGET_DIR)
	-mkdir -p $(TARGET_DIR)/data
	cp data/*.hdf $(TARGET_DIR)/data
	cp -r data/experimental_data $(TARGET_DIR)/data
	-mv $(CODE_DIR)/LICENSE $(TARGET_DIR)
	-mv $(CODE_DIR)/README.md $(TARGET_DIR)/CODE_README.md
	-mv $(CODE_DIR)/* $(TARGET_DIR)
	cp data/DATA_README.md $(TARGET_DIR)/DATA_README.md
	cd $(TARGET_DIR) && zip -Zstore -r ../supercurrent_data.zip *
