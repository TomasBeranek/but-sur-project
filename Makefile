TRAIN_TARGET_DIR=data/augment_target_train/
TEST_TARGET_DIR=data/target_dev/
TRAIN_NON_TARGET_DIR=data/augment_non_target_train/
TEST_NON_TARGET_DIR=data/non_target_dev/

# from each image in $(AUGMENT_SRC_DIR) will be created $(REPRODUCE_COEF) new images
REPRODUCE_COEF=20
AUGMENT_SRC_DIR=data/non_target_train
AUGMENT_DEST_DIR=data/augment_non_target_train

OPT_TRAIN=-v

all: test

test:
	python3 src/main.py $(TEST_NON_TARGET_DIR)

train:
	python3 src/main.py $(OPT_TRAIN) $(TRAIN_TARGET_DIR) $(TEST_TARGET_DIR) $(TRAIN_NON_TARGET_DIR) $(TEST_NON_TARGET_DIR)

augment:
	python3 src/data_augmentation.py $(AUGMENT_SRC_DIR) $(AUGMENT_DEST_DIR) $(REPRODUCE_COEF)
