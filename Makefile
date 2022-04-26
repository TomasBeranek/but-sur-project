TRAIN_TARGET_DIR=data/augment_target_train/
TEST_TARGET_DIR=data/target_dev/
TRAIN_NON_TARGET_DIR=data/augment_non_target_train/
TEST_NON_TARGET_DIR=data/non_target_dev/

# from each image in $(AUGMENT_SRC_DIR) will be created $(REPRODUCE_COEF) new images
REPRODUCE_COEF=6
AUGMENT_SRC_DIR=data/1_validate_target_train/
AUGMENT_DEST_DIR=data/1_validate_target_train_aug$(REPRODUCE_COEF)/

MODEL_DIR=models

1_CNN_FILE=1_cnn
2_CNN_FILE=2_cnn
3_CNN_FILE=3_cnn
GMM_FILE=gmm.json

OPT_TRAIN=--train
OPT_TEST =--test

TRAIN_T_AUG=_aug24
TRAIN_N_AUG=_aug4

all: test

test:
	python3 src/main.py $(OPT_TEST) data/1_validate_non_target_dev/ --cnn-file $(MODEL_DIR)/$(1_CNN_FILE)

train:
	python3 src/main.py $(OPT_TRAIN) $(TRAIN_TARGET_DIR) $(TEST_TARGET_DIR) $(TRAIN_NON_TARGET_DIR) $(TEST_NON_TARGET_DIR)

augment:
	mkdir $(AUGMENT_DEST_DIR)
	python3 src/data_augmentation.py $(AUGMENT_SRC_DIR) $(AUGMENT_DEST_DIR) $(REPRODUCE_COEF)

1-validate:
	python3 src/main.py $(OPT_TRAIN) data/1_validate_target_train$(TRAIN_T_AUG)/ data/1_validate_target_dev/ \
	data/1_validate_non_target_train$(TRAIN_N_AUG)/ data/1_validate_non_target_dev/ --cnn-file $(MODEL_DIR)/$(1_CNN_FILE) --gmm-file $(MODEL_DIR)/$(GMM_FILE)

2-validate:
	python3 src/main.py $(OPT_TRAIN) data/2_validate_target_train$(TRAIN_T_AUG)/ data/2_validate_target_dev/ \
	data/2_validate_non_target_train$(TRAIN_N_AUG)/ data/2_validate_non_target_dev/ --cnn-file $(MODEL_DIR)/$(2_CNN_FILE) --gmm-file $(MODEL_DIR)/$(GMM_FILE)

3-validate:
	python3 src/main.py $(OPT_TRAIN) data/3_validate_target_train$(TRAIN_T_AUG)/ data/3_validate_target_dev/ \
	data/3_validate_non_target_train$(TRAIN_N_AUG)/ data/3_validate_non_target_dev/ --cnn-file $(MODEL_DIR)/$(3_CNN_FILE) --gmm-file $(MODEL_DIR)/$(GMM_FILE)
