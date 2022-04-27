TRAIN_TARGET_DIR=data/augment_target_train/
TEST_TARGET_DIR=data/target_dev/
TRAIN_NON_TARGET_DIR=data/augment_non_target_train/
TEST_NON_TARGET_DIR=data/non_target_dev/

# from each image in $(AUGMENT_SRC_DIR) will be created $(REPRODUCE_COEF) new images
REPRODUCE_COEF=150
AUGMENT_SRC_DIR=data/all_target/
AUGMENT_DEST_DIR=data/all_validate_target_train_aug$(REPRODUCE_COEF)/

MODEL_DIR=models

1_CNN_FILE=$(MODEL_DIR)/1_cnn
2_CNN_FILE=$(MODEL_DIR)/2_cnn
3_CNN_FILE=$(MODEL_DIR)/3_cnn
CNN_FILE=$(MODEL_DIR)/cnn

1_GMM_FILE=$(MODEL_DIR)/1_gmm.json
2_GMM_FILE=$(MODEL_DIR)/2_gmm.json
3_GMM_FILE=$(MODEL_DIR)/3_gmm.json
GMM_FILE=$(MODEL_DIR)/gmm.json

OPT_TRAIN=--train
OPT_TEST =--test

TRAIN_T_AUG=_aug150
TRAIN_N_AUG=_aug25

all: test

test:
	python3 src/main.py $(OPT_TEST) data/2_validate_non_target_dev/ --cnn-file $(CNN_FILE) --gmm-file $(GMM_FILE)

train:
	python3 src/main.py $(OPT_TRAIN) $(TRAIN_TARGET_DIR) $(TEST_TARGET_DIR) $(TRAIN_NON_TARGET_DIR) $(TEST_NON_TARGET_DIR) --cnn-file $(CNN_FILE) --gmm-file $(GMM_FILE)

augment:
	mkdir $(AUGMENT_DEST_DIR)
	python3 src/data_augmentation.py $(AUGMENT_SRC_DIR) $(AUGMENT_DEST_DIR) $(REPRODUCE_COEF)

1-validate:
	python3 src/main.py $(OPT_TRAIN) data/1_validate_target_train$(TRAIN_T_AUG)/ data/1_validate_target_dev/ \
	data/1_validate_non_target_train$(TRAIN_N_AUG)/ data/1_validate_non_target_dev/ --cnn-file $(1_CNN_FILE) --gmm-file $(1_GMM_FILE)

2-validate:
	python3 src/main.py $(OPT_TRAIN) data/2_validate_target_train$(TRAIN_T_AUG)/ data/2_validate_target_dev/ \
	data/2_validate_non_target_train$(TRAIN_N_AUG)/ data/2_validate_non_target_dev/ --cnn-file $(2_CNN_FILE) --gmm-file $(2_GMM_FILE)

3-validate:
	python3 src/main.py $(OPT_TRAIN) data/3_validate_target_train$(TRAIN_T_AUG)/ data/3_validate_target_dev/ \
	data/3_validate_non_target_train$(TRAIN_N_AUG)/ data/3_validate_non_target_dev/ --cnn-file $(3_CNN_FILE) --gmm-file $(3_GMM_FILE)

all-validate:
	python3 src/main.py $(OPT_TRAIN) data/all_validate_target_train$(TRAIN_T_AUG)/ data/all_target/ \
	data/all_validate_non_target_train$(TRAIN_N_AUG)/ data/all_non_target/ --cnn-file $(CNN_FILE) --gmm-file $(GMM_FILE)

final:
	python3 src/main.py --final $(2_CNN_FILE) $(3_CNN_FILE) $(CNN_FILE) $(2_GMM_FILE) $(3_GMM_FILE) $(GMM_FILE) data/all_non_target/
